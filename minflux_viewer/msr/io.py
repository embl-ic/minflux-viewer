# minflux_msr/io.py
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import zarr

from .descriptions import describe_dtype, describe_path
from .utils import slug
from .state import reset as reset_state, set_mfx_for, set_mbm_for

# NOTE: ``specpy`` is an Abberior-supplied Windows-only binary wheel.
# We import it lazily so this module loads on every platform.
# The plugin UI catches ImportError and shows installation instructions.
specpy = None  # populated on first call to _ensure_specpy()


def _ensure_specpy():
    """Import specpy on demand; raise a user-readable error on failure."""
    global specpy
    if specpy is None:
        try:
            import specpy as _specpy
        except ImportError as exc:
            raise ImportError(
                "The 'specpy' library is required to parse .msr files.\n"
                "specpy is not on PyPI — obtain the matching wheel from your\n"
                "Imspector installation:\n"
                "  C:\\Imspector\\Versions\\<ver>\\python\\specpy\\\n"
                "Then install it:\n"
                "  poetry run pip install <path-to-wheel>.whl\n"
                "See INSTALL_MSR.md for full instructions."
            ) from exc
        specpy = _specpy
    return specpy

# -------- existing: collect_zarr_fields (unchanged) --------
def collect_zarr_fields(zroot: str) -> List[Dict[str, Any]]:
    g = zarr.open(zroot, mode="r")
    out: List[Dict[str, Any]] = []
    def visitor(path, obj):
        if path == "":
            return
        is_array = hasattr(obj, "shape") and hasattr(obj, "dtype")
        kind = "array" if is_array else "group"
        ent: Dict[str, Any] = {"path": path, "kind": kind}
        if is_array:
            ent["shape"] = tuple(getattr(obj, "shape", ()))
            ent["length"] = int(obj.shape[0]) if obj.shape else 0
            dt = getattr(obj, "dtype", "")
            ent["dtype_raw"] = str(dt)
            ent["dtype"] = describe_dtype(path, dt, ent["length"])
            ent["description"] = describe_path(path, is_array=True, dtype=dt)

            # Structured arrays expose child fields under the node.
            try:
                tree = _dtype_fields_tree(dt, ent["length"])
                if tree:
                    ent["dtype_fields"] = tree
                    if len(tree) == 1 and tree[0].get("kind") == "struct-root":
                        ent["dtype_singleton_root"] = tree[0]["name"]
            except Exception:
                pass
        else:
            ent["description"] = describe_path(path, is_array=False)
        out.append(ent)
    g.visititems(visitor)
    out.sort(key=lambda d: d["path"])
    return out

def read_zarr_attrs(zroot: str, path: str = "") -> dict:
    arch = zarr.open(zroot, mode="r")
    node = arch[path] if path else arch
    attrs = getattr(node, "attrs", None)
    if attrs is None:
        return {}
    try:
        return attrs.asdict()
    except Exception:
        try:
            return dict(attrs)
        except Exception:
            return {}


# -------- existing: parse_msr_to_tree (kept if you still use it elsewhere) --------

# -------- NEW: legacy helpers --------
def _logical_shape_str_for_field(n_rows: int, subshape: tuple) -> str:
    N = n_rows if (isinstance(n_rows, int) and n_rows > 0) else "N"
    if not subshape:
        return f"{N} × 1"
    dims = " × ".join(str(d) for d in subshape)
    return f"{dims} × {N}"

def _dtype_base_and_shape(dt: np.dtype):
    base = getattr(dt, "base", dt)
    shape = tuple(getattr(dt, "shape", ()) or ())
    return base, shape

def _dtype_fields_tree(dt: np.dtype, n_rows: int):
    """
    Recursively convert a structured dtype into a nested field tree.
    Each node: {name, dtype, shape, logical_shape, kind, children?}
    """
    names = getattr(dt, "names", None)
    if not names:
        return []
    out = []
    for name in names:
        
        sub_dt = dt.fields[name][0]
        base, subshape = _dtype_base_and_shape(sub_dt)
        node = {
            "name": name,
            "dtype": str(base),
            "shape": subshape,
            "logical_shape": _logical_shape_str_for_field(n_rows, subshape),
        }
        if getattr(base, "names", None):            # nested struct or structured subarray
            node["kind"] = "struct"
            node["children"] = _dtype_fields_tree(base, n_rows)
        else:
            node["kind"] = "field"
        out.append(node)
    # mark “singleton root” if there is exactly one field and it is a struct (e.g., itr)
    if len(out) == 1 and out[0].get("kind") == "struct":
        out[0]["kind"] = "struct-root"
    return out



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

def _try_load_legacy_mfx(msr_file: str, log=print):
    """Try to fetch MF/data {0} from legacy OBF-style MSR via msr-reader."""
    try:
        from msr_reader import OBFFile
    except Exception as e:
        log(f"[legacy] msr-reader not available: {e}")
        return None

    import numpy as np
    with OBFFile(msr_file) as obf:
        n = len(obf.shapes)
        target = None
        for i in range(n):
            try:
                xml = obf.get_imspector_xml_metadata(i)
            except Exception:
                xml = ""
            name = _series_name_from_xml(xml) or ""
            if "mf/data" in name.lower():
                target = i; break
        if target is None:
            # fallback: pick longest 1-D
            best_len = 0
            for i in range(n):
                a = np.asarray(obf.read_stack(i))
                if a.ndim == 1 and a.size > best_len:
                    best_len = a.size; target = i
        if target is None:
            return None
        arr = obf.read_stack(target)   # NumPy 1-D
        return arr

def _extract_legacy_reader_series(msr_file: str, log=print) -> dict:
    """
    Use msr-reader to enumerate legacy OBF stacks with a friendly 'StackSizes(...)' repr.
    Returns: {"count": int, "entries": [{"index": i, "name": str, "shape_repr": str}, ...]}
    """
    out = {"count": 0, "entries": []}
    try:
        from msr_reader import OBFFile
    except Exception as e:
        log(f"[legacy] msr-reader not available: {e}")
        return out

    with OBFFile(msr_file) as obf:
        shapes = getattr(obf, "shapes", [])
        n = len(shapes)
        out["count"] = n
        for i in range(n):
            try:
                xml = obf.get_imspector_xml_metadata(i)
            except Exception:
                xml = ""
            name = _series_name_from_xml(xml) or getattr(shapes[i], "name", f"Series_{i+1}")
            shape_repr = repr(shapes[i])  # typically "StackSizes(name='...', sizes=[...], ...)"
            out["entries"].append({
                "index": i,
                "name": name,
                "shape_repr": shape_repr,
            })
    return out



def _extract_legacy_series_from_metadata(meta) -> list:
    """
    Extract legacy 'series' from the OME-style metadata dict produced by specpy.File.metadata().
    Returns a list of dicts: {index, name, shape_str, dtype, kind}
    """
    out = []
    try:
        if not isinstance(meta, dict):
            return out
        ome_list = meta.get("OME")
        if not (isinstance(ome_list, list) and ome_list):
            return out
        ome0 = ome_list[0]
        root = ome0.get("") if isinstance(ome0, dict) else {}
        images = root.get("Image") or []
        for i, img in enumerate(images):
            inner = img.get("") if isinstance(img, dict) else {}
            name = inner.get("Name") or img.get("Name") or f"Series {i+1}"
            pixels_list = inner.get("Pixels") or []
            px = pixels_list[0] if pixels_list else {}
            pxd = px.get("") if isinstance(px, dict) else {}
            sx = pxd.get("SizeX", 0)
            sy = pxd.get("SizeY", 0)
            ptype = pxd.get("Type", "")
            # shape string like "1525 x 1275" or "668826202 x 1"
            shape_str = f"{sx} x {sy}" if sx and sy else f"{sx} x {sy}"
            # heuristic kind: wide/tall => image, 1D-ish => data
            kind = "image" if (isinstance(sx, int) and isinstance(sy, int) and sx > 1 and sy > 1) else "data"
            out.append({
                "index": i,
                "name": name,
                "shape_str": shape_str,
                "dtype": str(ptype),
                "kind": kind,
            })
    except Exception:
        # be permissive; just return what we have
        pass
    return out


def _dtypes_from_metadata(meta) -> list:
    """
    Extract per-series dtype strings from OME metadata produced by specpy.File.metadata().
    Returns a list dtype_by_index[i] = "uint8"/"int16"/"float32"/...
    """
    dtypes = []
    try:
        if not isinstance(meta, dict):
            return dtypes
        ome_list = meta.get("OME")
        if not (isinstance(ome_list, list) and ome_list):
            return dtypes
        ome0 = ome_list[0]
        root = ome0.get("") if isinstance(ome0, dict) else {}
        images = root.get("Image") or []
        for img in images:
            inner = img.get("") if isinstance(img, dict) else {}
            pixels_list = inner.get("Pixels") or []
            pxd = (pixels_list[0].get("") if pixels_list else {}) if isinstance(pixels_list[0], dict) else {}
            dt = pxd.get("Type", "")
            dtypes.append(str(dt) if dt is not None else "")
    except Exception:
        pass
    return dtypes


def _build_legacy_series_tree_entries(msr_file: str, meta) -> list[dict]:
    """
    Combine msr-reader 'shapes' (for sizes + internal name) and OME metadata (for dtype)
    to produce entries suitable for the UI tree.

    Each entry: { index, display_name, shape_str, dtype }
    """
    entries: list[dict] = []
    # dtype by index via metadata
    dtype_by_idx = _dtypes_from_metadata(meta)

    try:
        from msr_reader import OBFFile
    except Exception:
        # no msr-reader: we can still use metadata-only fallback
        # produce entries using OME Image/Name and SizeX/SizeY if present
        try:
            if not isinstance(meta, dict):
                return entries
            ome_list = meta.get("OME")
            if not (isinstance(ome_list, list) and ome_list):
                return entries
            images = (ome_list[0].get("") or {}).get("Image") or []
            for i, img in enumerate(images):
                inner = img.get("") if isinstance(img, dict) else {}
                name = inner.get("Name") or f"Series {i+1}"
                name_disp = str(name).replace("_", " ")
                pixels_list = inner.get("Pixels") or []
                pxd = (pixels_list[0].get("") if pixels_list else {}) if isinstance(pixels_list[0], dict) else {}
                sx, sy = pxd.get("SizeX", 0), pxd.get("SizeY", 0)
                if sx and sy:
                    shape_str = f"{sy} x {sx}"  # show as rows x cols
                else:
                    shape_str = f"{sx} x {sy}"
                dtype = dtype_by_idx[i] if i < len(dtype_by_idx) else ""
                entries.append({
                    "index": i,
                    "display_name": name_disp,
                    "shape_str": shape_str,
                    "dtype": dtype,
                })
            return entries
        except Exception:
            return entries

    # Full path with msr-reader
    with OBFFile(msr_file) as obf:
        shapes = getattr(obf, "shapes", [])
        for i, sh in enumerate(shapes):
            # name: prefer XML Name, else shape name
            try:
                xml = obf.get_imspector_xml_metadata(i)
            except Exception:
                xml = ""
            xml_name = _series_name_from_xml(xml)
            shape_name = getattr(sh, "name", f"Series_{i+1}")
            disp = (xml_name or shape_name or f"Series {i+1}").replace("_", " ")

            # sizes -> "rows x cols" (if 2D) else "N x 1"
            sizes = getattr(sh, "sizes", None)
            if isinstance(sizes, (list, tuple)) and len(sizes) >= 2:
                shape_str = f"{sizes[0]} x {sizes[1]}"
            elif isinstance(sizes, (list, tuple)) and len(sizes) == 1:
                shape_str = f"{sizes[0]} x 1"
            else:
                shape_str = str(sizes) if sizes is not None else ""

            dtype = dtype_by_idx[i] if i < len(dtype_by_idx) else ""
            entries.append({
                "index": i,
                "display_name": disp,
                "shape_str": shape_str,
                "dtype": dtype,
            })
    return entries



# -------- NEW: unified parse for UI --------
def parse_msr_general(msr_file: str, tmp_dir: str, log=print) -> Dict[str, Any]:
    from .msr_parser import parse_general

    return parse_general(msr_file, tmp_dir, log=log)

# --- Back-compat wrapper: prefer parse_msr_general() ---
def parse_msr_to_tree(msr_file: str, tmp_dir: str, log=print):
    """
    Back-compat shim. Old code imported parse_msr_to_tree; we now route to
    parse_msr_general and return its dict so imports stop breaking.

    Return value:
      dict with keys: "mode", "msr", and depending on mode:
        - modern:  "datasets": [ {did, display_name, zroot, fields}, ... ]
        - legacy:  "metadata": dict
    """
    return parse_msr_general(msr_file, tmp_dir, log=log)


# --- Back-compat: find a single .msr from a path (file or folder) ---
def pick_one_msr(input_path: Optional[str]) -> Optional[str]:
    """
    Return a single .msr path from `input_path`.
      - If input_path is an .msr file, return it.
      - If it's a directory, return the first *.msr inside (sorted).
      - If None/empty/invalid, return None.
    """
    if not input_path:
        return None
    try:
        p = Path(input_path)
    except Exception:
        return None

    if p.is_file() and p.suffix.lower() == ".msr":
        return str(p)

    if p.is_dir():
        # Pick the first *.msr (case-insensitive) in a stable order
        candidates = sorted(list(p.glob("*.msr")) + list(p.glob("*.MSR")))
        if candidates:
            return str(candidates[0])

    return None
