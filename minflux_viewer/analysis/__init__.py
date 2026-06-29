"""
minflux_viewer.analysis
========================
Numerical analysis routines surfaced as menu items.

The per-trace standard-deviation precision estimator, the Fourier Ring
Correlation (FRC) resolution estimator, and the MINFLUX Cramér-Rao bound
(CRLB) are all implemented.
"""

from .local_density import (
    compute_local_density_for_dataset,
    local_density_histogram_2d,
    local_density_kdtree,
    local_density_voxel_radius_count,
    run_local_density,
)
from .localization_precision import (
    crlb_precision,
    frc_resolution,
    run_crlb,
    run_frc,
    run_stddev_per_trace,
    stddev_per_trace,
)

__all__ = [
    "run_frc",
    "frc_resolution",
    "run_crlb",
    "crlb_precision",
    "run_stddev_per_trace",
    "stddev_per_trace",
    "compute_local_density_for_dataset",
    "local_density_histogram_2d",
    "local_density_kdtree",
    "local_density_voxel_radius_count",
    "run_local_density",
]
