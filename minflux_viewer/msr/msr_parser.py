from pathlib import Path
from typing import Any, Dict

import numpy as np
import zarr

from .parser_model import DatasetInfo, FieldInfo
from .state import reset as reset_state, set_mfx_for, set_mbm_for
from .utils import slug

# specpy is Windows-only — imported lazily so this module loads everywhere.
from .io import _ensure_specpy


def _try_import_specpy():
    """Return the specpy module, or ``None`` when it is unavailable.

    specpy is the preferred (vendor reference) reader but is Windows-only; when
    it is missing we fall back to the pure-Python ``msr-reader``.
    """
    try:
        return _ensure_specpy()
    except Exception:
        return None


def _logical_shape(n_rows: int, subshape: tuple) -> str:
    n = n_rows if isinstance(n_rows, int) and n_rows > 0 else "N"
    if not subshape:
        return f"{n} × 1"
    return " × ".join(str(d) for d in subshape) + f" × {n}"


def _dtype_fields(dt: np.dtype, n_rows: int) -> list[FieldInfo]:
    names = getattr(dt, "names", None)
    if not names:
        return []
    out: list[FieldInfo] = []
    for name in names:
        subdt = dt.fields[name][0]
        base = getattr(subdt, "base", subdt)
        shape = tuple(getattr(subdt, "shape", ()) or ())
        children = _dtype_fields(subdt, n_rows) if getattr(subdt, "names", None) else []
        out.append(
            FieldInfo(
                name=name,
                dtype=str(base),
                shape=shape,
                logical_shape=_logical_shape(n_rows, shape),
                kind="struct" if children else "field",
                children=children,
            )
        )
    if len(out) == 1 and out[0].kind == "struct":
        out[0].kind = "struct-root"
    return out


class ModernMSRParser:
    def parse(self, msr_path: str, tmp: str, log=print) -> list[DatasetInfo]:
        specpy = _ensure_specpy()
        sf = specpy.File(msr_path, specpy.File.Read)
        info = sf.minflux_datasets() or []
        result: list[DatasetInfo] = []
        for ds in info:
            did = ds.get("did")
            name = ds.get("name") or str(did)
            out_dir = Path(tmp) / slug(str(name))
            out_dir.mkdir(parents=True, exist_ok=True)
            sf.unpack(did, str(out_dir))
            grp = zarr.open(str(out_dir / "zarr"), mode="r")
            fields: list[FieldInfo] = []

            def traverse(node, path=""):
                if isinstance(node, zarr.Array):
                    n_rows = node.shape[0] if node.shape else 0
                    dtype = node.dtype
                    if dtype.names:
                        children = _dtype_fields(dtype, n_rows)
                        fields.append(
                            FieldInfo(
                                name=path,
                                dtype=str(dtype),
                                shape=node.shape,
                                logical_shape="",
                                kind="structured",
                                children=children,
                            )
                        )
                    else:
                        fields.append(
                            FieldInfo(
                                name=path,
                                dtype=str(dtype),
                                shape=node.shape,
                                logical_shape="",
                                kind="array",
                                children=[],
                            )
                        )
                elif isinstance(node, zarr.Group):
                    for k in node:
                        traverse(node[k], f"{path}/{k}" if path else k)

            traverse(grp)
            result.append(DatasetInfo(did=did, name=name, root=str(out_dir), fields=fields))
        return result


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
    def __init__(self):
        self.modern = ModernMSRParser()
        self.legacy = LegacyMSRParser()

    def parse(self, msr_file: str, tmp_dir: str, log=print) -> Dict[str, Any]:
        from .io import collect_zarr_fields

        reset_state()
        msr_path = Path(msr_file)
        out_root = Path(tmp_dir) / f"msr_{msr_path.stem}"
        out_root.mkdir(parents=True, exist_ok=True)

        specpy = _try_import_specpy()
        if specpy is None:
            log("[parse] specpy unavailable — using pure-Python msr-reader")
            return self._parse_with_msr_reader(msr_file, out_root, collect_zarr_fields, log)

        sf = specpy.File(str(msr_file), specpy.File.Read)

        try:
            info = sf.minflux_datasets() or []
        except Exception as e:
            log(f"[warn] minflux_datasets() failed: {e}")
            info = []

        if info:
            log(f"[parse] modern MINFLUX file: {len(info)} dataset(s)")
            datasets = []
            for i, ds in enumerate(info):
                did = ds.get("did")
                name = ds.get("name") or ds.get("label") or str(did)
                key = str(name)
                ds_dir = out_root / slug(str(name))
                ds_dir.mkdir(parents=True, exist_ok=True)
                log(f"  ds#{i} did={did} name={name} -> {ds_dir}")
                sf.unpack(did, str(ds_dir))
                zroot = ds_dir / "zarr"
                fields = collect_zarr_fields(str(zroot)) if zroot.is_dir() else []
                if zroot.is_dir():
                    try:
                        arch = zarr.open(str(zroot), mode="r")
                        if "mfx" in arch:
                            arr = arch["mfx"][:]
                            set_mfx_for(key, arr)
                            log(f"    [mfx] loaded: key='{key}' shape={arr.shape} dtype={arr.dtype}")
                        if "grd/mbm/points" in arch:
                            arr = arch["grd/mbm/points"][:]
                            set_mbm_for(key, arr)
                            log(f"    [mbm] loaded: key='{key}' shape={arr.shape} dtype={arr.dtype}")
                    except Exception as e:
                        log(f"    [warn] array load failed: {e}")

                datasets.append(
                    {
                        "did": str(did),
                        "display_name": name,
                        "zroot": str(zroot),
                        "fields": fields,
                    }
                )
            return {"mode": "modern", "msr": str(msr_file), "datasets": datasets}

        # Early "obf / mfxdta" files: no minflux_datasets(), but the MINFLUX data
        # is embedded as a uint8 MFXDTA stack wrapping a zarr store. Unpack it to
        # disk so the rest of the modern pipeline (tree, mfx_map, "Open in MINFLUX
        # viewer") works unchanged.
        mfxdta_result = self._parse_mfxdta(sf, msr_file, out_root, collect_zarr_fields, log)
        if mfxdta_result is not None:
            return mfxdta_result

        log("[parse] legacy/OBF-style file (no minflux_datasets)")
        try:
            meta = sf.metadata()
            log("OME-XML metadata loaded; showing in UI")
        except Exception as e:
            meta = None
            log(f"[warn] metadata() failed: {e}")

        legacy_series_tree = self.legacy.build_series_tree_entries(str(msr_file), meta)
        return {
            "mode": "legacy",
            "msr": str(msr_file),
            "metadata": meta,
            "legacy_series_tree": legacy_series_tree,
        }

    def _mfxdta_dataset_entry(self, stack_idx, desc, blob, out_root, collect_zarr_fields, log):
        """Unpack one MFXDTA blob to a zarr store and return a modern-style
        dataset dict (also registering its ``mfx`` array), or ``None`` on error.

        Shared by the specpy and msr-reader paths so the unpack/register logic
        lives in exactly one place. MBM/bead data is intentionally skipped.
        """
        from .mfxdta import (
            SOURCE_FORMAT,
            container_version,
            extract_zarr_store,
            unpack_zarr_store_to_dir,
        )

        key = str(desc)
        zroot = out_root / slug(key) / "zarr"
        try:
            unpack_zarr_store_to_dir(extract_zarr_store(blob), zroot)
        except Exception as e:
            log(f"  [warn] stack {stack_idx}: failed to unpack MFXDTA store: {e}")
            return None
        fields = collect_zarr_fields(str(zroot)) if zroot.is_dir() else []
        try:
            arch = zarr.open(str(zroot), mode="r")
            if "mfx" not in arch:
                log(f"  [warn] stack {stack_idx}: no 'mfx' array after unpack")
                return None
            arr = arch["mfx"][:]
            set_mfx_for(key, arr)
            log(f"    [mfx] loaded key='{key}' shape={arr.shape} dtype={arr.dtype}")
        except Exception as e:
            log(f"    [warn] stack {stack_idx}: mfx load failed: {e}")
            return None
        return {
            "did": f"mfxdta:{stack_idx}",
            "display_name": desc,
            "zroot": str(zroot),
            "fields": fields,
            "source_format": SOURCE_FORMAT,
            "mfxdta_version": container_version(blob),
        }

    def _mfxdta_result(self, stacks, msr_file, out_root, collect_zarr_fields, log):
        """Build a ``mode="modern"`` result from MFXDTA ``(idx, desc, blob)``
        stacks, or ``None`` when nothing could be unpacked."""
        from .mfxdta import SOURCE_FORMAT

        datasets = [
            entry
            for idx, desc, blob in stacks
            if (entry := self._mfxdta_dataset_entry(
                idx, desc, blob, out_root, collect_zarr_fields, log)) is not None
        ]
        if not datasets:
            return None
        return {
            "mode": "modern",
            "msr": str(msr_file),
            "datasets": datasets,
            "source_format": SOURCE_FORMAT,
        }

    def _parse_mfxdta(self, sf, msr_file, out_root, collect_zarr_fields, log):
        """Decode embedded ``obf / mfxdta`` stacks via specpy (early format)."""
        from .mfxdta import SOURCE_FORMAT, find_mfxdta_stacks

        stacks = find_mfxdta_stacks(sf)
        if not stacks:
            return None
        log(f"[parse] {SOURCE_FORMAT}: {len(stacks)} embedded MINFLUX dataset(s)")
        result = self._mfxdta_result(stacks, msr_file, out_root, collect_zarr_fields, log)
        if result is None:
            log(f"[parse] {SOURCE_FORMAT} detected but nothing could be unpacked")
        return result

    def _parse_with_msr_reader(self, msr_file, out_root, collect_zarr_fields, log):
        """Specpy-free parse via msr-reader: recover MINFLUX data from MFXDTA
        stacks (early + modern), else build the legacy image-series tree."""
        from .mfxdta import SOURCE_FORMAT, read_obf_mfxdta_stacks

        try:
            stacks = read_obf_mfxdta_stacks(msr_file)
        except Exception as e:
            log(f"[warn] msr-reader could not read stacks: {e}")
            stacks = []
        if stacks:
            log(f"[parse] {SOURCE_FORMAT} (msr-reader): {len(stacks)} MINFLUX dataset(s)")
            result = self._mfxdta_result(stacks, msr_file, out_root, collect_zarr_fields, log)
            if result is not None:
                return result

        log("[parse] no MINFLUX data — building legacy image-series tree (msr-reader)")
        return {
            "mode": "legacy",
            "msr": str(msr_file),
            "metadata": None,
            "legacy_series_tree": self.legacy.build_series_tree_entries(str(msr_file), meta=None),
        }


def parse_modern(msr_path: str, tmp: str, log=print) -> list[DatasetInfo]:
    return ModernMSRParser().parse(msr_path, tmp, log=log)


def parse_legacy(msr_path: str, tmp: str, log=print) -> list[DatasetInfo]:
    # keep old signature, produce lightweight DatasetInfo compatibility structure
    series = LegacyMSRParser().build_series_tree_entries(msr_path, meta={})
    fields = [
        FieldInfo(
            name=str(s.get("display_name", "series")),
            dtype=str(s.get("dtype", "")),
            shape=tuple(),
            logical_shape=s.get("shape_str", ""),
            kind="legacy",
            children=[],
        )
        for s in series
    ]
    return [DatasetInfo(did=msr_path, name=Path(msr_path).stem, root="", fields=fields)]


def parse_general(msr_file: str, tmp_dir: str, log=print) -> Dict[str, Any]:
    return GeneralMSRParser().parse(msr_file, tmp_dir, log=log)


def flatten(dt):
    return []