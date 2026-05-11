# minflux_msr/export.py
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional, Tuple

import numpy as np

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def make_export_base(export_root: str, did: str, idx: int) -> str:
    """Kept for back-compat with your older flows."""
    out_root = Path(export_root); ensure_dir(out_root)
    return str(out_root / f"data#{idx}_(did#{did})")

# ---------- writers ----------

def _as_column(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2:
        return a
    return a.reshape(a.shape[0], -1)

def _mat_write_fast(out_path: Path, mfx: Optional[np.ndarray], mbm: Optional[np.ndarray], log=print):
    """Fast .mat: explode structured fields into plain arrays (v7 via scipy)."""
    try:
        from scipy.io import savemat
        saver = lambda p, obj: savemat(p, obj, do_compression=False, oned_as='column', long_field_names=True)
    except Exception as e:
        log(f"[warn] SciPy not available; skipping .mat ({e})")
        return None

    payload: Dict[str, Any] = {}

    def add_struct(prefix: str, arr: np.ndarray):
        names = getattr(arr.dtype, "names", None)
        if names:
            for k in names:
                col = arr[k]
                subnames = getattr(col.dtype, "names", None)
                if subnames:
                    for sk in subnames:
                        payload[f"{k}_{sk}"] = _as_column(col[sk])
                else:
                    payload[k] = _as_column(col)
        else:
            payload[prefix] = _as_column(arr)

    if mfx is not None:
        add_struct("mfx", mfx)
    if mbm is not None:
        # mbm usually plain or small-structured; store as-is (columns)
        payload["mbm"] = _as_column(mbm)

    if not payload:
        log("[skip] nothing to write to .mat"); return None

    try:
        saver(str(out_path), payload)
        log(f"[mat] wrote {out_path}")
        return str(out_path)
    except Exception as e:
        log(f"[error] .mat save failed: {e}")
        return None

def _npy_write(out_path: Path, name: str, arr: Optional[np.ndarray], log=print):
    if arr is None:
        return None
    fn = out_path.with_name(out_path.stem + f"_{name}.npy")
    try:
        np.save(str(fn), arr, allow_pickle=False)
        log(f"[npy] wrote {fn}")
        return str(fn)
    except Exception as e:
        log(f"[error] .npy save failed for {name}: {e}")
        return None

def _npz_write(out_path: Path, mfx: Optional[np.ndarray], mbm: Optional[np.ndarray], log=print):
    fn = out_path.with_suffix(".npz")
    try:
        d = {}
        if mfx is not None: d["mfx"] = mfx
        if mbm is not None: d["mbm"] = mbm
        if not d:
            log("[skip] nothing to write to .npz"); return None
        np.savez_compressed(str(fn), **d)
        log(f"[npz] wrote {fn}")
        return str(fn)
    except Exception as e:
        log(f"[error] .npz save failed: {e}")
        return None

# JSON fast (chunk)
_WANTED_FIELDS = [
    "vld","fnl","bot","eot",
    "sta","tim","tid","gri","thi","sqi","itr",
    "loc","lnc","eco","ecc","efo","efc","fbg","cfr","dcr"
]

def _rows_to_dicts(records, fields_present):
    out = []
    for r in records:
        d = {}
        for k, v in zip(fields_present, r):
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, np.generic):
                d[k] = v.item()
            else:
                d[k] = v
        out.append(d)
    return out

def _json_write(out_path: Path, mfx: Optional[np.ndarray], chunk_rows: int, log=print):
    if mfx is None:
        return None
    try:
        import orjson
        dumps = lambda obj: orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY)
    except Exception:
        import json as _json
        dumps = lambda obj: _json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    names = getattr(mfx.dtype, "names", None)
    if not names:
        log("[json] mfx has no named fields; skipping JSON."); return None
    fields_present = [k for k in _WANTED_FIELDS if k in names]

    fn = out_path.with_suffix(".json")
    N = int(mfx.shape[0])
    with open(fn, "wb") as fh:
        fh.write(b"[\n")
        first = True
        for start in range(0, N, chunk_rows):
            stop = min(start + chunk_rows, N)
            sub = mfx[start:stop][fields_present]
            recs = sub.tolist()
            objs = _rows_to_dicts(recs, fields_present)
            payload = dumps(objs)
            if payload.startswith(b"[") and payload.endswith(b"]"):
                payload = payload[1:-1]
            if not first and payload:
                fh.write(b",\n")
            fh.write(payload)
            first = False
        fh.write(b"\n]\n")
    log(f"[json] wrote {fn}")
    return str(fn)

# ---------- public: export using arrays ----------
def export_arrays(out_dir: str,
                  base_name: str,
                  formats: List[str],
                  mfx: Optional[np.ndarray],
                  mbm: Optional[np.ndarray],
                  log=print,
                  json_chunk_rows: int = 100_000) -> List[str]:
    """
    Write selected formats to out_dir with base_name.
    formats: any of ["mat", "npy", "npz", "json", "csv"]
    Returns list of written file paths.
    """
    out_root = Path(out_dir); ensure_dir(out_root)
    out_base = out_root / base_name
    written: List[str] = []

    if "mat" in formats:
        p = _mat_write_fast(out_base.with_suffix(".mat"), mfx, mbm, log=log)
        if p: written.append(p)
    if "npy" in formats:
        p1 = _npy_write(out_base, "mfx", mfx, log=log);  p2 = _npy_write(out_base, "mbm", mbm, log=log)
        written.extend([p for p in (p1, p2) if p])
    if "npz" in formats:
        p = _npz_write(out_base, mfx, mbm, log=log)
        if p: written.append(p)
    if "json" in formats:
        p = _json_write(out_base, mfx, chunk_rows=json_chunk_rows, log=log)
        if p: written.append(p)
    if "csv" in formats:
        log("[info] CSV of full structured arrays is not supported here. "
            "Use 'Fields included…' to export selected fields to CSV.")

    return written

# ---------- optional: existing API (kept) ----------
def export_selected_fields(zroot: str, did: str, paths: List[str], base: str, formats: List[str], log=print):
    # (keep your previous implementation)
    log("[todo] export_selected_fields: using your existing code path.")
