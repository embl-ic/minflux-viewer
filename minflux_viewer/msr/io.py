# minflux_msr/io.py
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import zarr

from .descriptions import describe_dtype, describe_path

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
