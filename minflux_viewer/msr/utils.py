import re

import numpy as np


def slug(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9._\-#()]+", "_", s)  # allow # ( )
    return s[:120] or "untitled"

def to_numpy(arr_like):
    if arr_like is None:
        return None
    if isinstance(arr_like, np.ndarray):
        return arr_like
    try:
        values = arr_like[:]  # zarr arrays
    except TypeError:  # an array protocol without slice support
        return np.array(arr_like)
    # I/O, allocation and conversion failures are not missing slice support.
    return np.asarray(values)

def to_2d(a: np.ndarray) -> np.ndarray:
    if a.ndim <= 2:
        return a
    return a.reshape(a.shape[0], -1)
