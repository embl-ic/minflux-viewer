"""
minflux_viewer.analysis
========================
Numerical analysis routines surfaced as menu items.

For Phase 4 only the per-trace standard-deviation precision estimator is
implemented; FRC and CRLB are exposed as information dialogs that cite
their reference papers and are slated for a dedicated implementation
session.
"""

from .localization_precision import (
    show_frc_info,
    show_crlb_info,
    run_stddev_per_trace,
    stddev_per_trace,
)
from .local_density import (
    compute_local_density_for_dataset,
    local_density_histogram_2d,
    local_density_kdtree,
    run_local_density,
)

__all__ = [
    "show_frc_info",
    "show_crlb_info",
    "run_stddev_per_trace",
    "stddev_per_trace",
    "compute_local_density_for_dataset",
    "local_density_histogram_2d",
    "local_density_kdtree",
    "run_local_density",
]
