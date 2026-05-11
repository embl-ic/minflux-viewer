# minflux_msr/state.py
from typing import Optional, Dict
import numpy as np

# Back-compat singletons (first dataset loaded)
mfx: Optional[np.ndarray] = None
mbm: Optional[np.ndarray] = None

# New: hold multiple datasets, keyed by dataset label (or DID fallback)
mfx_map: Dict[str, np.ndarray] = {}
mbm_map: Dict[str, np.ndarray] = {}

def reset():
    """Clear all global data holders."""
    global mfx, mbm, mfx_map, mbm_map
    mfx = None
    mbm = None
    mfx_map = {}
    mbm_map = {}

def set_mfx_for(key: str, arr: np.ndarray):
    """Store MFX array for a dataset key; keep first one also at state.mfx."""
    global mfx, mfx_map
    if key is None:
        key = "dataset"
    mfx_map[key] = arr
    if mfx is None:
        mfx = arr

def set_mbm_for(key: str, arr: np.ndarray):
    """Store MBM array for a dataset key; keep first one also at state.mbm."""
    global mbm, mbm_map
    if key is None:
        key = "dataset"
    mbm_map[key] = arr
    if mbm is None:
        mbm = arr
