from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import zarr

from .parser_model import DatasetInfo, FieldInfo
from .state import reset as reset_state, set_mfx_for, set_mbm_for
from .utils import slug

# specpy is Windows-only — imported lazily so this module loads everywhere.
from .io import _ensure_specpy


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
    @staticmethod
    def _series_name_from_xml(xml_text: str) -> Optional[str]:
        try:
            import xml.etree.ElementTree as ET

            root = ET.fromstring(xml_text)
            for el in root.iter():
                tag = el.tag.rsplit("}", 1)[-1].lower()
                if tag == "name" and el.text:
                    return el.text.strip()
        except Exception:
            pass
        return None

    @staticmethod
    def _dtypes_from_metadata(meta) -> list[str]:
        dtypes: list[str] = []
        try:
            if not isinstance(meta, dict):
                return dtypes
            ome_list = meta.get("OME")
            if not (isinstance(ome_list, list) and ome_list):
                return dtypes
            images = ((ome_list[0] or {}).get("") or {}).get("Image") or []
            for img in images:
                inner = img.get("") if isinstance(img, dict) else {}
                pixels_list = inner.get("Pixels") or []
                pxd = (pixels_list[0].get("") if pixels_list else {}) if isinstance(pixels_list[0], dict) else {}
                dtypes.append(str(pxd.get("Type", "")))
        except Exception:
            pass
        return dtypes

    def build_series_tree_entries(self, msr_file: str, meta) -> list[dict]:
        entries: list[dict] = []
        dtype_by_idx = self._dtypes_from_metadata(meta)
        try:
            from msr_reader import OBFFile
        except Exception:
            return entries

        with OBFFile(msr_file) as obf:
            shapes = getattr(obf, "shapes", [])
            for i, sh in enumerate(shapes):
                try:
                    xml = obf.get_imspector_xml_metadata(i)
                except Exception:
                    xml = ""
                xml_name = self._series_name_from_xml(xml)
                shape_name = getattr(sh, "name", f"Series_{i+1}")
                disp = (xml_name or shape_name or f"Series {i+1}").replace("_", " ")
                sizes = getattr(sh, "sizes", None)
                if isinstance(sizes, (list, tuple)) and len(sizes) >= 2:
                    shape_str = f"{sizes[0]} x {sizes[1]}"
                elif isinstance(sizes, (list, tuple)) and len(sizes) == 1:
                    shape_str = f"{sizes[0]} x 1"
                else:
                    shape_str = str(sizes) if sizes is not None else ""
                entries.append(
                    {
                        "index": i,
                        "display_name": disp,
                        "shape_str": shape_str,
                        "dtype": dtype_by_idx[i] if i < len(dtype_by_idx) else "",
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
        specpy = _ensure_specpy()
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