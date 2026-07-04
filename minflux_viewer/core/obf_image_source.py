"""
OBF (``.msr``) image source — a lazy, series-aware image reader that presents
Abberior OBF image stacks through the **same interface as** :class:`~minflux_viewer.core.tiff_source.TiffImageSource`,
so the standalone TIFF viewer can display image series parsed from a ``.msr`` file
(the ones Fiji's Bio-Formats importer lists as loadable series).

An ``.msr`` embeds one OBF stack per "series". Some stacks are genuine 2-D/3-D
images (confocal/overview channels); others are 1-D histograms/populations, and
MINFLUX data is a 1-D ``uint8`` MFXDTA blob. Only the ``ndim >= 2`` stacks are
exposed here as viewable image series.

Pure helpers (``axes_for_sizes``, ``classify_image_only``) are unit-tested without
a file; :class:`ObfImageSource` wraps ``msr_reader.OBFFile`` and reuses
``TiffMetadata`` so no viewer-side metadata changes are needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Importing the msr package applies the OBF reader compat monkeypatches
# (meta_data_position == 0 sentinel), needed for some legacy files to open.
import minflux_viewer.msr  # noqa: F401

from .tiff_source import PhysicalSize, TiffMetadata


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


def classify_image_only(stacks: list[dict]) -> bool:
    """Is a ``.msr`` **image-only** (no MINFLUX data), given per-stack header info?

    *stacks* is ``[{"ndim": int, "dtype": str}, …]``. MINFLUX data is always a
    1-D ``uint8`` MFXDTA blob, so a file with **no** 1-D ``uint8`` stack but at
    least one ``ndim >= 2`` image stack is image-only. Header-only (no pixel reads),
    so it stays cheap even for large MINFLUX files."""
    has_image = any(int(s.get("ndim", 0)) >= 2 for s in stacks)
    has_mfx = any(int(s.get("ndim", 0)) == 1 and str(s.get("dtype", "")) == "uint8" for s in stacks)
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


def _scan_stacks(path) -> list[dict]:
    """Header-only scan of every OBF stack: ``sizes``/``ndim``/``dtype``/``name``.
    No pixel data is read."""
    out: list[dict] = []
    with _open_obf(path) as obf:
        for i in range(obf.num_stacks):
            sizes = tuple(int(s) for s in (obf.shapes[i].sizes or []))
            try:
                pixel = tuple(float(p) for p in (obf.pixel_sizes[i].sizes or []))
            except Exception:
                pixel = ()
            header = obf.stack_headers[i]
            name = (getattr(header, "description", "") or getattr(header, "name", "")
                    or f"Series {i + 1}")
            out.append({
                "raw_index": i,
                "name": str(name).replace("_", " ").strip() or f"Series {i + 1}",
                "sizes": sizes,
                "ndim": len(sizes),
                "pixel_m": pixel,
                "dtype": _stack_dtype_name(header),
            })
    return out


def msr_is_image_only(path) -> bool:
    """True when *path* is an image-only ``.msr`` (route it to the image viewer)."""
    try:
        return classify_image_only(_scan_stacks(path))
    except Exception:
        return False


def list_obf_image_series(path) -> list[dict]:
    """The viewable (``ndim >= 2``) image series of a ``.msr``, in file order:
    ``[{"raw_index", "name", "shape_str", "dtype"}, …]``."""
    series = []
    for s in _scan_stacks(path):
        if s["ndim"] >= 2:
            series.append({
                "raw_index": s["raw_index"],
                "name": s["name"],
                "shape_str": " x ".join(str(v) for v in s["sizes"]),
                "dtype": s["dtype"],
            })
    return series


class ObfImageSource:
    """Lazy, series-aware image reader for a ``.msr`` OBF file.

    Mirrors :class:`~minflux_viewer.core.tiff_source.TiffImageSource`: exposes a
    ``TiffMetadata``, ``axis_size``, ``read_plane`` and ``close``, plus series
    helpers (``series_names`` / ``series_summaries`` / ``set_series``). Series are
    the ``ndim >= 2`` stacks; ``series_index`` addresses that filtered list,
    ``raw_stack_index`` addresses the underlying OBF stack."""

    def __init__(self, path, *, series_index: int = 0, raw_stack_index: int | None = None) -> None:
        self.path = Path(path)
        self._obf = _open_obf(self.path)
        self._stacks = [s for s in _scan_stacks(self.path) if s["ndim"] >= 2]
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

    def _load_array(self) -> np.ndarray:
        if self._array is None:
            raw = self._stacks[self.series_index]["raw_index"]
            self._array = np.asarray(self._obf.read_stack(raw))
        return self._array

    def read_plane(self, *, t: int = 0, c: int = 0, z: int = 0) -> np.ndarray:
        arr = self._load_array()
        axes = self.metadata.axes
        z = int(np.clip(z, 0, self.axis_size("Z") - 1))
        if "Z" in axes and arr.ndim >= 3:
            return np.asarray(arr[z])
        return np.asarray(arr)

    def close(self) -> None:
        try:
            self._obf.close()
        except Exception:
            pass

    # -- metadata -------------------------------------------------------------
    def _build_metadata(self) -> TiffMetadata:
        s = self._stacks[self.series_index]
        sizes = tuple(int(v) for v in s["sizes"])
        axes = axes_for_sizes(sizes)
        pixel = list(s["pixel_m"]) + [None] * 3
        # pixel[k] aligns to axis k of the (numpy-order) sizes / array.
        pz = _nm(pixel[axes.index("Z")]) if "Z" in axes else PhysicalSize(None, "nm")
        py = _nm(pixel[axes.index("Y")]) if "Y" in axes else PhysicalSize(None, "nm")
        px = _nm(pixel[axes.index("X")]) if "X" in axes else PhysicalSize(None, "nm")
        dtype = s["dtype"] or "unknown"
        ome_xml = self._ome_xml()
        summary = [
            ("File", str(self.path)),
            ("Series", f"{self.series_index + 1} / {len(self._stacks)}  (OBF stack {s['raw_index']})"),
            ("Image name", s["name"]),
            ("Axes", axes),
            ("Shape", " x ".join(str(v) for v in sizes)),
            ("dtype", dtype),
            ("Source", "OBF / .msr"),
            ("Pixel size X", px.display()),
            ("Pixel size Y", py.display()),
            ("Pixel size Z", pz.display()),
        ]
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
            channel_names=(),
            image_name=s["name"],
            raw_summary=tuple(summary),
            ome_xml=ome_xml,
        )

    def _ome_xml(self) -> str | None:
        try:
            xml = self._obf.get_ome_xml_metadata()
            return str(xml) if xml else None
        except Exception:
            return None
