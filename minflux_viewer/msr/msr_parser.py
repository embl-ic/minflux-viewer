import contextlib
import logging
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from ..utils import zarr2
from .state import reset as reset_state
from .state import set_mbm_for, set_mbm_meta_for, set_mfx_for

#: Expected failures of the pixel-free OBF stack scan (as image_series_of).
_SCAN_ERRORS = (OSError, IndexError, KeyError, TypeError, ValueError, EOFError)


def _ome_xml_metadata(msr_file: str, obf=None) -> str | None:
    """Return the file-level OME-XML document exactly as stored in OBF.

    *obf* is an already-open ``OBFFile`` of *msr_file* to read it from, so
    the parse does not open the file again; it is left open."""
    try:
        from msr_reader import OBFFile

        opened = contextlib.nullcontext(obf) if obf is not None else OBFFile(msr_file)
        with opened as handle:
            xml = handle.get_ome_xml_metadata()
        return str(xml) if xml else None
    except MemoryError:
        raise
    except Exception:
        # Optional metadata crosses an opaque third-party decoder boundary.
        logging.getLogger(__name__).exception("Could not decode OBF OME metadata")
        return None


def _ome_metadata(msr_file: str, xml: str | None = None) -> dict | None:
    """Return the OBF/OME metadata as a dict via msr-reader, or ``None``.

    Used for the legacy ``<metadata>`` tree of image-only ``.msr`` files (those
    without MINFLUX data).
    """
    try:
        xml = xml if xml is not None else _ome_xml_metadata(msr_file)
        if not xml:
            return None
        return {"OME": _xml_to_dict(ET.fromstring(xml))}
    except (ET.ParseError, TypeError, ValueError, RecursionError):
        return None


def _xml_to_dict(el) -> Any:
    """Generic ElementTree → nested dict (tag, attributes, children, text)."""
    node: dict[str, Any] = {}
    node.update({f"@{k}": v for k, v in el.attrib.items()})
    for child in el:
        tag = child.tag.rsplit("}", 1)[-1]
        val = _xml_to_dict(child)
        if tag in node:
            existing = node[tag]
            node[tag] = existing if isinstance(existing, list) else [existing]
            node[tag].append(val)
        else:
            node[tag] = val
    text = (el.text or "").strip()
    if text and not node:
        return text
    if text:
        node["#text"] = text
    return node


class LegacyMSRParser:
    """Image-series tree for OBF/legacy ``.msr`` files (no MINFLUX data).

    Names, sizes and dtypes come straight from ``msr-reader``'s OBF stack
    headers, so no hand-written XML/OME parsing is needed.
    """

    @staticmethod
    def _dtype_name(dtype_flags: int) -> str:
        try:
            from msr_reader.obffile import _parse_dtype

            return np.dtype(_parse_dtype(int(dtype_flags))[0]).name
        except (ImportError, KeyError, TypeError, ValueError, OverflowError):
            return ""

    def build_series_tree_entries(self, msr_file: str, meta=None, log=print,
                                  obf=None) -> list[dict]:
        """*obf*: an already-open ``OBFFile`` of *msr_file* (left open)."""
        entries: list[dict] = []
        try:
            from msr_reader import OBFFile
        except ImportError:
            return entries

        # Defense-in-depth: a malformed/variant OBF header must not crash the open
        # (the patched reader handles meta_data_position==0; this catches the rest).
        try:
            opened = contextlib.nullcontext(obf) if obf is not None else OBFFile(msr_file)
            with opened as obf:
                for i in range(obf.num_stacks):
                    header = obf.stack_headers[i]
                    sh = obf.shapes[i]
                    name = (
                        getattr(header, "description", "")
                        or getattr(header, "name", "")
                        or getattr(sh, "name", "")
                        or f"Series {i + 1}"
                    )
                    sizes = list(getattr(sh, "sizes", []) or [])
                    shape_str = " x ".join(str(s) for s in sizes) if sizes else ""
                    entries.append(
                        {
                            "index": i,
                            # Verbatim: the stack name is what Imspector shows
                            # and what an image export is named after.
                            "display_name": str(name),
                            "shape_str": shape_str,
                            "dtype": self._dtype_name(getattr(header, "dtype", 0)),
                        }
                    )
        except MemoryError:
            raise
        except Exception as exc:  # third-party OBF decoding; report, do not crash a batch
            try:
                log(f"[warn] could not read OBF image-series tree: {exc}")
            except Exception:  # caller-supplied logging must not mask the decoder error
                logging.getLogger(__name__).exception("OBF series warning callback failed")
        return entries


class GeneralMSRParser:
    """Parse an ``.msr`` file with the pure-Python ``msr-reader`` (no specpy)."""

    def __init__(self):
        self.legacy = LegacyMSRParser()

    def parse(self, msr_file: str, tmp_dir=None, log=print) -> dict[str, Any]:
        """Recover MINFLUX data from the embedded MFXDTA stacks (early single-
        channel + modern multi-channel). Files with no MINFLUX data fall back to
        a legacy image-series tree.

        Decoding is fully in-memory — each dataset's zarr store is kept as a
        ``{key: bytes}`` dict, not written to disk (``tmp_dir`` is unused, kept
        only for call-site compatibility). Writing a store to disk is now an
        explicit ``.zarr`` export option, not a parse-time cache.

        **The file is opened and read once.** Its OBF headers give the dataset
        labels, the per-stack ``minflux`` tags and the image-series list; only
        the ``uint8`` payload stacks are read beyond that, each in one large
        read, inflated and decoded on threads leased from the process budget
        (:func:`minflux_viewer.core.execution.acquire_cpu_threads`). What later steps
        would otherwise re-read from the file travels with the result:

        * ``did_label_map`` -- dataset ``did`` → label;
        * ``acquisition_rois`` -- the MINFLUX acquisition rectangles, from each
          dataset store's root ``.zattrs`` (compressed stacks included);
        * ``obf_stacks`` -- the pixel-free scan of every stack
          (:func:`~minflux_viewer.core.obf_image_source.scan_obf`), ``None``
          when that scan failed, and ``image_series`` -- the viewable ones.
        """
        from msr_reader import OBFFile

        from ..core.execution import acquire_cpu_threads
        from ..core.obf_image_source import image_series_from_scan, scan_obf
        from .io import collect_zarr_fields
        from .mfxdta import (
            SOURCE_FORMAT,
            minflux_stack_tags_from_obf,
            read_mfxdta_stacks_from_obf,
        )

        reset_state()

        obf = None
        # Threads for inflating and decoding, leased from the process budget
        # for the whole parse and given back however it ends.
        lease = acquire_cpu_threads()
        try:
            try:
                obf = OBFFile(str(msr_file))
                stacks = read_mfxdta_stacks_from_obf(obf, lease=lease)
            except MemoryError:
                raise
            except Exception as e:  # report third-party decoding failures
                log(f"[warn] msr-reader could not read stacks: {e}")
                stacks = []

            obf_stacks = None
            if obf is not None:
                try:
                    obf_stacks = scan_obf(obf)
                except _SCAN_ERRORS as e:
                    log(f"[image] could not list image series: {e}")
            scan = {
                "obf_stacks": obf_stacks,
                "image_series": (None if obf_stacks is None
                                 else image_series_from_scan(obf_stacks)),
            }

            if stacks:
                result = self._modern_result(
                    msr_file, obf, stacks, lease, collect_zarr_fields,
                    minflux_stack_tags_from_obf(obf), SOURCE_FORMAT, log)
                if result is not None:
                    result.update(scan)
                    return result
                log(f"[parse] {SOURCE_FORMAT} detected but could not be decoded")

            log("[parse] no MINFLUX data — building legacy image-series tree")
            metadata_xml = _ome_xml_metadata(msr_file, obf=obf)
            return {
                "mode": "legacy",
                "msr": str(msr_file),
                "metadata": _ome_metadata(msr_file, metadata_xml),
                "metadata_xml": metadata_xml,
                "legacy_series_tree": self.legacy.build_series_tree_entries(
                    str(msr_file), log=log, obf=obf),
                **scan,
            }
        finally:
            lease.release()
            if obf is not None:
                obf.close()

    def _modern_result(self, msr_file, obf, stacks, lease, collect_zarr_fields,
                       stack_tags, source_format, log) -> dict[str, Any] | None:
        """The ``mode="modern"`` result for the decoded MFXDTA *stacks*, or
        ``None`` when none of them decodes."""
        from ..core.acquisition_metadata import (
            did_label_map_from_obf,
            rois_from_attrs,
            unique_rois,
        )

        log(f"[parse] {source_format}: {len(stacks)} MINFLUX dataset(s)")
        label_map = did_label_map_from_obf(obf)
        with zarr2.decode_threads(lease):
            datasets = [
                entry
                for idx, desc, blob in stacks
                if (entry := self._mfxdta_dataset_entry(
                    idx, desc, blob, collect_zarr_fields, label_map,
                    stack_tags.get(idx, {}), log)) is not None
            ]
        if not datasets:
            return None
        # Per-parse maps (name -> array) assembled from the entries, so a
        # reader dialog can keep its own copy independent of the shared
        # module-global ``state`` (which only holds the last parse).
        mfx_map = {d["display_name"]: d["_mfx"] for d in datasets if d.get("_mfx") is not None}
        mbm_map = {d["display_name"]: d["_mbm"] for d in datasets if d.get("_mbm") is not None}
        mbm_meta_map = {d["display_name"]: d["_mbm_meta"] for d in datasets if d.get("_mbm_meta")}
        rois = unique_rois(
            roi for d in datasets
            for roi in rois_from_attrs((d.get("zroot") or {}).get(".zattrs") or b"{}"))
        return {
            "mode": "modern",
            "msr": str(msr_file),
            "datasets": datasets,
            "source_format": source_format,
            "mfx_map": mfx_map,
            "mbm_map": mbm_map,
            "mbm_meta_map": mbm_meta_map,
            "did_label_map": label_map,
            "acquisition_rois": rois,
        }

    def _mfxdta_dataset_entry(
        self, stack_idx, desc, blob, collect_zarr_fields, label_map,
        stack_tag, log,
    ):
        """Decode one MFXDTA blob to an in-memory zarr store, register its
        ``mfx`` (and any ``grd/mbm/points`` beads), and return a modern-style
        dataset dict — or ``None`` on error. ``zroot`` holds the in-memory store
        (a ``{key: bytes}`` dict that ``zarr2.open`` reads directly). Channel name
        = the file's did→label, else the stack description, else a synthetic name.
        """
        from .mfxdta import SOURCE_FORMAT, container_version, extract_zarr_store

        try:
            store = extract_zarr_store(blob)            # {key: bytes} — no disk
            arch = zarr2.open(store, mode="r")
        except MemoryError:
            raise
        except Exception as e:  # malformed per-stack containers are reported and skipped
            log(f"  [warn] stack {stack_idx}: failed to decode MFXDTA store: {e}")
            return None
        if "mfx" not in arch:
            log(f"  [warn] stack {stack_idx}: no 'mfx' array in store")
            return None
        mbm_arr = None
        mbm_meta = None
        try:
            mfx_node = arch["mfx"]
            attrs_did = str(getattr(mfx_node, "attrs", {}).get("did", "") or "")
            tag_did = str((stack_tag or {}).get("did", "") or "")
            did = attrs_did or tag_did
            tag_label = str((stack_tag or {}).get("label", "") or "")
            name = label_map.get(did) or tag_label or str(desc) or f"minflux_{stack_idx}"
            arr = mfx_node[:]
            log(f"    [mfx] loaded key='{name}' shape={arr.shape} dtype={arr.dtype}")
            if "grd/mbm/points" in arch:
                beads = arch["grd/mbm/points"][:]
                if getattr(beads, "size", 0):
                    mbm_arr = beads
                    log(f"    [mbm] loaded key='{name}' shape={beads.shape}")
            else:
                # Legacy bead layout: nested grd/mbm/<R-ID> structured arrays.
                # Translate to the model points array + metadata (tree unchanged).
                from .legacy_mbm import build_legacy_mbm
                legacy = build_legacy_mbm(arch, log)
                if legacy is not None:
                    pts, pbg, used = legacy
                    if getattr(pts, "size", 0):
                        mbm_arr = pts
                    mbm_meta = {"points_by_gri": pbg, "used": used}
            entry = {
                "did": did or f"mfxdta:{stack_idx}",
                "obf_stack_index": int(stack_idx),
                "display_name": name,
                "zroot": store,
                "fields": collect_zarr_fields(store),
                "source_format": SOURCE_FORMAT,
                "mfxdta_version": container_version(blob),
                # Per-parse arrays belong to this reader, not the global aliases.
                "_mfx": arr,
                "_mbm": mbm_arr,
                "_mbm_meta": mbm_meta,
            }
        except MemoryError:
            raise
        except Exception as e:  # report a failed stack without discarding other stacks
            log(f"    [warn] stack {stack_idx}: mfx load failed: {e}")
            return None
        # Publish only after the entire stack (including beads and metadata) is
        # decoded. A failed bead chunk must not leave a half-loaded channel.
        set_mfx_for(name, arr)
        if mbm_arr is not None:
            set_mbm_for(name, mbm_arr)
        if mbm_meta is not None:
            set_mbm_meta_for(name, mbm_meta)
        return entry


def parse_general(msr_file: str, tmp_dir: str, log=print) -> dict[str, Any]:
    return GeneralMSRParser().parse(msr_file, tmp_dir, log=log)
