from typing import Any, Dict

import numpy as np
import zarr

from .state import reset as reset_state, set_mfx_for, set_mbm_for


def _ome_metadata(msr_file: str) -> dict | None:
    """Return the OBF/OME metadata as a dict via msr-reader, or ``None``.

    Used for the legacy ``<metadata>`` tree of image-only ``.msr`` files (those
    without MINFLUX data).
    """
    try:
        import xml.etree.ElementTree as ET

        from msr_reader import OBFFile

        with OBFFile(msr_file) as obf:
            xml = obf.get_ome_xml_metadata()
        if not xml:
            return None
        return {"OME": _xml_to_dict(ET.fromstring(xml))}
    except Exception:
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
        except Exception:
            return ""

    def build_series_tree_entries(self, msr_file: str, meta=None) -> list[dict]:
        entries: list[dict] = []
        try:
            from msr_reader import OBFFile
        except Exception:
            return entries

        with OBFFile(msr_file) as obf:
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
                        "display_name": str(name).replace("_", " "),
                        "shape_str": shape_str,
                        "dtype": self._dtype_name(getattr(header, "dtype", 0)),
                    }
                )
        return entries


class GeneralMSRParser:
    """Parse an ``.msr`` file with the pure-Python ``msr-reader`` (no specpy)."""

    def __init__(self):
        self.legacy = LegacyMSRParser()

    def parse(self, msr_file: str, tmp_dir=None, log=print) -> Dict[str, Any]:
        """Recover MINFLUX data from the embedded MFXDTA stacks (early single-
        channel + modern multi-channel). Files with no MINFLUX data fall back to
        a legacy image-series tree.

        Decoding is fully in-memory — each dataset's zarr store is kept as a
        ``{key: bytes}`` dict, not written to disk (``tmp_dir`` is unused, kept
        only for call-site compatibility). Writing a store to disk is now an
        explicit ``.zarr`` export option, not a parse-time cache.
        """
        from .io import collect_zarr_fields
        from .mfxdta import SOURCE_FORMAT, extract_did_label_map, read_obf_mfxdta_stacks

        reset_state()

        try:
            stacks = read_obf_mfxdta_stacks(msr_file)
        except Exception as e:
            log(f"[warn] msr-reader could not read stacks: {e}")
            stacks = []

        if stacks:
            log(f"[parse] {SOURCE_FORMAT}: {len(stacks)} MINFLUX dataset(s)")
            label_map = extract_did_label_map(msr_file)
            datasets = [
                entry
                for idx, desc, blob in stacks
                if (entry := self._mfxdta_dataset_entry(
                    idx, desc, blob, collect_zarr_fields, label_map, log)) is not None
            ]
            if datasets:
                return {
                    "mode": "modern",
                    "msr": str(msr_file),
                    "datasets": datasets,
                    "source_format": SOURCE_FORMAT,
                }
            log(f"[parse] {SOURCE_FORMAT} detected but could not be decoded")

        log("[parse] no MINFLUX data — building legacy image-series tree")
        return {
            "mode": "legacy",
            "msr": str(msr_file),
            "metadata": _ome_metadata(msr_file),
            "legacy_series_tree": self.legacy.build_series_tree_entries(str(msr_file)),
        }

    def _mfxdta_dataset_entry(self, stack_idx, desc, blob, collect_zarr_fields, label_map, log):
        """Decode one MFXDTA blob to an in-memory zarr store, register its
        ``mfx`` (and any ``grd/mbm/points`` beads), and return a modern-style
        dataset dict — or ``None`` on error. ``zroot`` holds the in-memory store
        (a ``{key: bytes}`` dict that ``zarr.open`` reads directly). Channel name
        = the file's did→label, else the stack description, else a synthetic name.
        """
        from .mfxdta import SOURCE_FORMAT, container_version, extract_zarr_store

        try:
            store = extract_zarr_store(blob)            # {key: bytes} — no disk
            arch = zarr.open(store, mode="r")
        except Exception as e:
            log(f"  [warn] stack {stack_idx}: failed to decode MFXDTA store: {e}")
            return None
        if "mfx" not in arch:
            log(f"  [warn] stack {stack_idx}: no 'mfx' array in store")
            return None
        try:
            mfx_node = arch["mfx"]
            did = str(getattr(mfx_node, "attrs", {}).get("did", "") or "")
            name = label_map.get(did) or str(desc) or f"minflux_{stack_idx}"
            arr = mfx_node[:]
            set_mfx_for(name, arr)
            log(f"    [mfx] loaded key='{name}' shape={arr.shape} dtype={arr.dtype}")
            if "grd/mbm/points" in arch:
                beads = arch["grd/mbm/points"][:]
                if getattr(beads, "size", 0):
                    set_mbm_for(name, beads)
                    log(f"    [mbm] loaded key='{name}' shape={beads.shape}")
        except Exception as e:
            log(f"    [warn] stack {stack_idx}: mfx load failed: {e}")
            return None
        return {
            "did": did or f"mfxdta:{stack_idx}",
            "display_name": name,
            "zroot": store,                             # in-memory zarr store
            "fields": collect_zarr_fields(store),
            "source_format": SOURCE_FORMAT,
            "mfxdta_version": container_version(blob),
        }


def parse_general(msr_file: str, tmp_dir: str, log=print) -> Dict[str, Any]:
    return GeneralMSRParser().parse(msr_file, tmp_dir, log=log)