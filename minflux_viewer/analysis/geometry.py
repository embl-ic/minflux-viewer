"""
minflux_viewer.analysis.geometry
================================
Small shared geometry helpers for the analysis modules (kept in one place so the
same routine isn't re-implemented per module).
"""

from __future__ import annotations

import numpy as np


def rotation_a_to_b(a, b) -> np.ndarray:
    """Rotation matrix taking unit-ish vector *a* onto *b* (Rodrigues formula).

    Handles the (anti)parallel degenerate cases: identity when already aligned,
    ``-I`` when antiparallel.
    """
    a = np.asarray(a, float); a = a / (np.linalg.norm(a) or 1.0)
    b = np.asarray(b, float); b = b / (np.linalg.norm(b) or 1.0)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(a @ b)
    if s < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))
