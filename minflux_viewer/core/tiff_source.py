"""
Lazy TIFF image source for standalone TIFF viewer windows.

This module intentionally stays outside the MINFLUX dataset model.  TIFF files
opened through File > Open .tif File... are image-viewer sessions, not
localization datasets.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

from .tiff_roi import read_roi_from_tiff

SOURCE_METADATA_NAMESPACE = "https://minflux-viewer.org/ome/source-metadata/1.0"
SOURCE_OME_XML_KEY = "OriginalOMEImageXML"
SOURCE_IMSPECTOR_XML_KEY = "ImspectorXML"
SOURCE_OBF_JSON_KEY = "OBFStackMetadataJSON"
SOURCE_MINFLUX_JSON_KEY = "MINFLUXStackTagJSON"


@dataclass(frozen=True)
class PhysicalSize:
    value: float | None
    unit: str

    @property
    def nm(self) -> float | None:
        if self.value is None:
            return None
        unit = (self.unit or "").strip().lower().replace("µ", "u")
        scale = {
            "nm": 1.0,
            "nanometer": 1.0,
            "nanometers": 1.0,
            "um": 1000.0,
            "micrometer": 1000.0,
            "micrometers": 1000.0,
            "micron": 1000.0,
            "microns": 1000.0,
            "mm": 1_000_000.0,
            "cm": 10_000_000.0,
            "inch": 25_400_000.0,
            "in": 25_400_000.0,
            "m": 1_000_000_000.0,
        }.get(unit)
        if scale is None:
            return None
        return float(self.value) * scale

    def display(self) -> str:
        if self.value is None:
            return "unknown"
        return f"{self.value:g} {self.unit or ''}".strip()


@dataclass(frozen=True)
class MetadataDocument:
    """One source metadata document attached to an image series."""

    name: str
    format: str
    content: str


@dataclass(frozen=True)
class TiffMetadata:
    path: Path
    series_index: int
    series_count: int
    axes: str
    shape: tuple[int, ...]
    dtype: str
    is_ome: bool
    is_imagej: bool
    pixel_size_x: PhysicalSize
    pixel_size_y: PhysicalSize
    pixel_size_z: PhysicalSize
    channel_names: tuple[str, ...]
    image_name: str
    raw_summary: tuple[tuple[str, str], ...]
    ome_xml: str | None = None
    acquisition_date: str = ""
    description: str = ""
    time_interval: PhysicalSize = field(default_factory=lambda: PhysicalSize(None, "s"))
    documents: tuple[MetadataDocument, ...] = ()

    def axis_size(self, axis: str) -> int:
        if axis not in self.axes:
            return 1
        return int(self.shape[self.axes.index(axis)])

    @property
    def pixel_size_x_nm(self) -> float:
        return float(self.pixel_size_x.nm or 1.0)

    @property
    def pixel_size_y_nm(self) -> float:
        return float(self.pixel_size_y.nm or 1.0)


class TiffImageSource:
    """Metadata-aware, lazy-plane TIFF reader backed by tifffile."""

    def __init__(self, path: str | Path, *, series_index: int = 0) -> None:
        try:
            import tifffile
        except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
            raise ImportError("tifffile is required to open TIFF images") from exc

        self.path = Path(path)
        self._tifffile = tifffile
        self._tf = tifffile.TiffFile(str(self.path))
        if not self._tf.series:
            self.close()
            raise ValueError(f"'{self.path.name}' contains no readable TIFF image series")
        if series_index < 0 or series_index >= len(self._tf.series):
            self.close()
            raise IndexError(f"TIFF series index {series_index} is out of range")
        self.series_index = int(series_index)
        self._series = self._tf.series[self.series_index]
        self.metadata = self._build_metadata()

    def close(self) -> None:
        try:
            self._tf.close()
        except Exception:
            pass

    # -- series ---------------------------------------------------------------
    @property
    def series_count(self) -> int:
        return len(self._tf.series)

    def series_names(self) -> list[str]:
        return [
            str(getattr(s, "name", "") or f"Series {i + 1}")
            for i, s in enumerate(self._tf.series)
        ]

    def series_summaries(self) -> list[dict]:
        """One row per series for the series chooser: index, name, shape, dtype."""
        out: list[dict] = []
        for i, s in enumerate(self._tf.series):
            shape = tuple(int(v) for v in getattr(s, "shape", ()) or ())
            out.append({
                "index": i,
                "name": str(getattr(s, "name", "") or f"Series {i + 1}"),
                "shape_str": " x ".join(str(v) for v in shape),
                "dtype": str(getattr(s, "dtype", "")),
            })
        return out

    def set_series(self, index: int) -> None:
        """Switch to another series in the same file (rebuilds metadata)."""
        index = int(index)
        if not (0 <= index < len(self._tf.series)):
            raise IndexError(f"TIFF series index {index} is out of range")
        self.series_index = index
        self._series = self._tf.series[self.series_index]
        self.metadata = self._build_metadata()

    def axis_size(self, axis: str) -> int:
        return self.metadata.axis_size(axis)

    def active_roi(self):
        """The image's ImageJ active ROI (:class:`TiffRoi`), or ``None``."""
        return self._roi

    def read_plane(self, *, t: int = 0, c: int = 0, z: int = 0) -> np.ndarray:
        """Read one 2-D plane lazily, returning YX or YXS data."""
        axes = self.metadata.axes
        shape = self.metadata.shape
        indices_by_axis = {"T": int(t), "C": int(c), "Z": int(z)}
        for axis, value in list(indices_by_axis.items()):
            indices_by_axis[axis] = int(np.clip(value, 0, self.axis_size(axis) - 1))

        page = self._read_plane_page(indices_by_axis)
        if page is not None:
            return self._normalize_plane(np.asarray(page), axes_hint=None)

        arr = self._series.asarray()
        selection: list[int | slice] = []
        kept_axes: list[str] = []
        for axis, size in zip(axes, shape):
            if axis in {"Y", "X", "S"}:
                selection.append(slice(None))
                kept_axes.append(axis)
            elif axis in indices_by_axis:
                selection.append(indices_by_axis[axis])
            else:
                selection.append(0)
        plane = np.asarray(arr[tuple(selection)])
        return self._normalize_plane(plane, axes_hint="".join(kept_axes))

    def _read_plane_page(self, indices_by_axis: dict[str, int]) -> np.ndarray | None:
        axes = self.metadata.axes
        shape = self.metadata.shape
        prefix_axes: list[str] = []
        prefix_shape: list[int] = []
        prefix_indices: list[int] = []
        for axis, size in zip(axes, shape):
            if axis in {"Y", "X", "S"}:
                continue
            prefix_axes.append(axis)
            prefix_shape.append(int(size))
            prefix_indices.append(int(indices_by_axis.get(axis, 0)))
        if not prefix_shape:
            index = 0
        else:
            try:
                index = int(np.ravel_multi_index(tuple(prefix_indices), tuple(prefix_shape)))
            except Exception:
                return None

        pages = self._series.pages
        try:
            if len(pages) <= index:
                return None
            return pages[index].asarray()
        except Exception:
            return None

    @staticmethod
    def _normalize_plane(plane: np.ndarray, axes_hint: str | None) -> np.ndarray:
        arr = np.asarray(plane)
        arr = np.squeeze(arr)
        if arr.ndim < 2:
            return np.reshape(arr, (1, max(int(arr.size), 1)))
        if axes_hint:
            axes = axes_hint
            target = [axes.index("Y"), axes.index("X")]
            if "S" in axes:
                target.append(axes.index("S"))
            if len(target) == arr.ndim:
                arr = np.transpose(arr, target)
        if arr.ndim > 3:
            arr = np.squeeze(arr)
        if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
            arr = np.nanmax(arr.astype(np.float64, copy=False), axis=0)
        return arr

    def _build_metadata(self) -> TiffMetadata:
        series = self._series
        axes = str(getattr(series, "axes", "") or "")
        shape = tuple(int(v) for v in getattr(series, "shape", ()) or ())
        if not axes or len(axes) != len(shape):
            axes = self._infer_axes(shape)
        axes = self._canonical_axes(axes, shape)
        dtype = str(getattr(series, "dtype", "unknown"))
        ome_xml = getattr(self._tf, "ome_metadata", None)
        imagej_metadata = getattr(self._tf, "imagej_metadata", None) or {}
        # The ImageJ active-ROI tag is a per-file block, so it belongs to the
        # first series; a multi-series file cannot address a ROI per series.
        self._roi = read_roi_from_tiff(self._tf) if self.series_index == 0 else None
        series_ome_xml = extract_ome_image_xml(ome_xml, self.series_index)
        ome_info = _parse_ome_metadata(series_ome_xml, 0)
        imagej_info = _parse_imagej_metadata(imagej_metadata)
        tiff_info = _parse_tiff_resolution(series)

        px = (ome_info.get("pixel_size_x") or imagej_info.get("pixel_size_x")
              or tiff_info.get("pixel_size_x") or PhysicalSize(None, ""))
        py = (ome_info.get("pixel_size_y") or imagej_info.get("pixel_size_y")
              or tiff_info.get("pixel_size_y") or PhysicalSize(None, ""))
        pz = ome_info.get("pixel_size_z") or imagej_info.get("pixel_size_z") or PhysicalSize(None, "")
        time_interval = (
            ome_info.get("time_interval")
            or imagej_info.get("time_interval")
            or PhysicalSize(None, "s")
        )
        acquisition_date = str(
            ome_info.get("acquisition_date") or tiff_info.get("acquisition_date") or ""
        )
        channel_names = tuple(ome_info.get("channel_names") or ())
        image_name = str(
            ome_info.get("image_name")
            or getattr(series, "name", "")
            or self.path.stem
        )

        summary = [
            ("File", str(self.path)),
            ("Series", f"{self.series_index + 1} / {len(self._tf.series)}"),
            ("Image name", image_name or self.path.stem),
            ("Axes", axes),
            ("Shape", " x ".join(str(v) for v in shape)),
            ("dtype", dtype),
            ("OME-TIFF", "yes" if ome_xml else "no"),
            ("ImageJ metadata", "yes" if imagej_metadata else "no"),
            ("Pixel size X", px.display()),
            ("Pixel size Y", py.display()),
            ("Pixel size Z", pz.display()),
        ]
        if channel_names:
            summary.append(("Channels", ", ".join(channel_names)))
        if acquisition_date:
            summary.append(("Acquisition date", acquisition_date))
        if time_interval.value is not None:
            summary.append(("Frame interval", time_interval.display()))
        summary.append(("Active ROI", self._roi.summary() if self._roi else "none"))
        documents: list[MetadataDocument] = []
        if series_ome_xml:
            documents.append(MetadataDocument("OME-XML", "xml", series_ome_xml))
        documents.extend(_source_documents_from_ome(ome_info))
        if imagej_metadata:
            documents.append(MetadataDocument(
                "ImageJ metadata", "json",
                json.dumps(imagej_metadata, indent=2, ensure_ascii=False, default=str),
            ))
        header = _tiff_header_document(series)
        if header:
            documents.append(MetadataDocument("TIFF header", "json", header))
        return TiffMetadata(
            path=self.path,
            series_index=self.series_index,
            series_count=len(self._tf.series),
            axes=axes,
            shape=shape,
            dtype=dtype,
            is_ome=bool(ome_xml),
            is_imagej=bool(imagej_metadata),
            pixel_size_x=px,
            pixel_size_y=py,
            pixel_size_z=pz,
            channel_names=channel_names,
            image_name=image_name,
            raw_summary=tuple(summary),
            ome_xml=series_ome_xml,
            acquisition_date=acquisition_date,
            description=str(ome_info.get("description") or ""),
            time_interval=time_interval,
            documents=tuple(documents),
        )

    @staticmethod
    def _infer_axes(shape: tuple[int, ...]) -> str:
        if len(shape) == 2:
            return "YX"
        if len(shape) == 3 and shape[-1] in (3, 4):
            return "YXS"
        if len(shape) == 3:
            return "ZYX"
        if len(shape) == 4 and shape[-1] in (3, 4):
            return "ZYXS"
        if len(shape) == 4:
            return "CZYX"
        if len(shape) == 5:
            return "TCZYX"
        labels = "ABCDEFGHIJKLMNOPQRSTUVW"
        return "".join(labels[i] if i < len(labels) else "Q" for i in range(len(shape)))

    @staticmethod
    def _canonical_axes(axes: str, shape: tuple[int, ...]) -> str:
        """Map tifffile's generic page axis to viewer-friendly Z when needed."""
        if len(axes) != len(shape):
            return axes
        chars = list(axes)
        for idx, axis in enumerate(chars):
            if axis != "S":
                continue
            is_trailing_color_samples = idx == len(chars) - 1 and int(shape[idx]) in (3, 4)
            if not is_trailing_color_samples:
                chars[idx] = "Z" if "Z" not in chars else "Q"
        return "".join(chars)


def extract_ome_image_xml(ome_xml: str | None, series_index: int) -> str | None:
    """Return a self-contained OME document for exactly one image series.

    OBF stores one file-level OME document containing every stack. Passing that
    whole document to a per-image viewer made Ch1 appear to own Ch2's metadata.
    The selected ``Image`` and only its referenced structured annotations are
    retained here.
    """
    if not ome_xml:
        return None
    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError:
        return None
    images = [child for child in list(root) if _local_name(child.tag) == "Image"]
    if not images:
        images = [el for el in root.iter() if _local_name(el.tag) == "Image"]
    if not images:
        return None
    selected = images[min(max(int(series_index), 0), len(images) - 1)]
    selected_copy = deepcopy(selected)

    for child in list(root):
        if _local_name(child.tag) == "Image":
            root.remove(child)
    insert_at = next(
        (i for i, child in enumerate(list(root)) if _local_name(child.tag) == "StructuredAnnotations"),
        len(root),
    )
    root.insert(insert_at, selected_copy)

    referenced = {
        str(el.attrib.get("ID") or "")
        for el in selected_copy.iter()
        if _local_name(el.tag) == "AnnotationRef"
    }
    for annotations_el in [el for el in root if _local_name(el.tag) == "StructuredAnnotations"]:
        for annotation in list(annotations_el):
            if str(annotation.attrib.get("ID") or "") not in referenced:
                annotations_el.remove(annotation)
        if len(annotations_el) == 0:
            root.remove(annotations_el)
    try:
        ET.indent(root, space="  ")
    except AttributeError:  # pragma: no cover - Python < 3.9
        pass
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _parse_ome_metadata(ome_xml: str | None, series_index: int) -> dict[str, Any]:
    if not ome_xml:
        return {}
    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError:
        return {}
    images = [el for el in root.iter() if _local_name(el.tag) == "Image"]
    if not images:
        return {}
    image_el = images[min(max(int(series_index), 0), len(images) - 1)]
    image = _strip_ns(image_el)
    pixels_el = next((el for el in image_el.iter() if _local_name(el.tag) == "Pixels"), None)
    if pixels_el is None:
        return {}
    pixels = _strip_ns(pixels_el)
    channel_els = [
        el
        for el in pixels_el.iter()
        if _local_name(el.tag) == "Channel"
    ]
    channels = [_strip_ns(el) for el in channel_els]
    channel_names = []
    for idx, ch in enumerate(channels):
        name = ch.get("Name") or ch.get("ID") or f"C{idx + 1}"
        channel_names.append(str(name))
    result: dict[str, Any] = {
        "image_name": image.get("Name") or image.get("ID"),
        "pixel_size_x": _physical_size(pixels, "PhysicalSizeX", "PhysicalSizeXUnit"),
        "pixel_size_y": _physical_size(pixels, "PhysicalSizeY", "PhysicalSizeYUnit"),
        "pixel_size_z": _physical_size(pixels, "PhysicalSizeZ", "PhysicalSizeZUnit"),
        "channel_names": tuple(channel_names),
        "acquisition_date": next(
            ((el.text or "").strip() for el in image_el
             if _local_name(el.tag) == "AcquisitionDate"),
            "",
        ),
        "description": next(
            ((el.text or "").strip() for el in image_el
             if _local_name(el.tag) == "Description"),
            "",
        ),
        "source_annotations": _ome_map_annotations(root),
        "sizes": {
            axis: int(pixels.get(f"Size{axis}", "1"))
            for axis in "TCZYX"
            if str(pixels.get(f"Size{axis}", "")).isdigit()
        },
        "dimension_order": str(pixels.get("DimensionOrder") or ""),
    }
    time_increment = _physical_size(pixels, "TimeIncrement", "TimeIncrementUnit")
    if time_increment is not None:
        result["time_interval"] = time_increment
    return result


def _parse_imagej_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    unit = str(metadata.get("unit") or "")
    spacing = metadata.get("spacing")
    if spacing is not None:
        try:
            result["pixel_size_z"] = PhysicalSize(float(spacing), unit)
        except Exception:
            pass
    interval = metadata.get("finterval")
    if interval is None:
        try:
            fps = float(metadata.get("fps"))
            interval = 1.0 / fps if fps > 0 else None
        except Exception:
            interval = None
    if interval is not None:
        try:
            result["time_interval"] = PhysicalSize(float(interval), "s")
        except Exception:
            pass
    return result


def _parse_tiff_resolution(series) -> dict[str, Any]:
    """Read calibration and acquisition time from standard TIFF tags."""
    try:
        page = series.pages[0]
        tags = page.tags
    except Exception:
        return {}
    result: dict[str, Any] = {}
    try:
        result["acquisition_date"] = str(tags["DateTime"].value)
    except Exception:
        pass
    try:
        x_res = _resolution_value(tags["XResolution"].value)
        y_res = _resolution_value(tags["YResolution"].value)
        raw_unit = tags["ResolutionUnit"].value
        unit_name = str(getattr(raw_unit, "name", raw_unit)).upper()
    except Exception:
        return result
    if x_res <= 0 or y_res <= 0:
        return result
    if unit_name in {"3", "CENTIMETER"}:
        unit = "cm"
    elif unit_name in {"2", "INCH"}:
        unit = "inch"
    else:
        # ResolutionUnit=NONE is commonly paired with ImageJ's unit string.
        try:
            unit = str((page.parent.imagej_metadata or {}).get("unit") or "")
        except Exception:
            unit = ""
    if not unit:
        return result
    result.update({
        "pixel_size_x": PhysicalSize(1.0 / x_res, unit),
        "pixel_size_y": PhysicalSize(1.0 / y_res, unit),
    })
    return result


def _resolution_value(value) -> float:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return float(value[0]) / float(value[1])
    return float(value)


def _ome_map_annotations(root) -> dict[str, str]:
    result: dict[str, str] = {}
    for annotation in root.iter():
        if _local_name(annotation.tag) != "MapAnnotation":
            continue
        namespace = str(annotation.attrib.get("Namespace") or "")
        if namespace and namespace != SOURCE_METADATA_NAMESPACE:
            continue
        for entry in annotation.iter():
            if _local_name(entry.tag) != "M":
                continue
            key = str(entry.attrib.get("K") or "")
            if key:
                result[key] = entry.text or ""
    return result


def _source_documents_from_ome(info: dict[str, Any]) -> list[MetadataDocument]:
    annotations = info.get("source_annotations") or {}
    labels = {
        SOURCE_OME_XML_KEY: ("Original OME-XML", "xml"),
        SOURCE_IMSPECTOR_XML_KEY: ("Imspector XML", "xml"),
        SOURCE_OBF_JSON_KEY: ("OBF stack metadata", "json"),
        SOURCE_MINFLUX_JSON_KEY: ("MINFLUX stack tag", "json"),
    }
    return [
        MetadataDocument(labels[key][0], labels[key][1], str(value))
        for key, value in annotations.items()
        if key in labels and value
    ]


def _tiff_header_document(series) -> str:
    """Readable JSON representation of the first page's TIFF IFD tags."""
    try:
        page = series.pages[0]
    except Exception:
        return ""
    values: dict[str, Any] = {}
    for tag in page.tags.values():
        name = str(getattr(tag, "name", tag.code))
        value = getattr(tag, "value", None)
        if name == "ImageDescription" and isinstance(value, str) and len(value) > 4000:
            values[name] = f"<{len(value)} characters; see the metadata documents>"
        elif isinstance(value, (bytes, bytearray)):
            values[name] = f"<{len(value)} bytes>"
        elif isinstance(value, np.ndarray):
            values[name] = value.tolist() if value.size <= 64 else f"<array shape={value.shape}>"
        else:
            values[name] = value
    return json.dumps(values, indent=2, ensure_ascii=False, default=str)


def _physical_size(attrs: dict[str, str], value_key: str, unit_key: str) -> PhysicalSize | None:
    value = attrs.get(value_key)
    if value is None:
        return None
    try:
        return PhysicalSize(float(value), str(attrs.get(unit_key) or "um"))
    except Exception:
        return None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _strip_ns(element) -> dict[str, str]:
    return {str(k).rsplit("}", 1)[-1]: str(v) for k, v in element.attrib.items()}
