"""Canonical bead-point columns shared by readers and analyses."""

import numpy as np

POINTS_DTYPE = np.dtype([
    ("gri", np.int64), ("xyz", np.float64, (3,)),
    ("tim", np.float64), ("str", np.float64),
])
