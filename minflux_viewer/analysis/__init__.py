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
from .tracking_adapters import (
    TrackingStateAdapterSpec,
    get_tracking_state_adapter,
    register_tracking_state_adapter,
    run_tracking_state_adapter,
    tracking_state_adapter_availability,
    tracking_state_adapters,
)
from .tracking_advanced import (
    confinement_index,
    ergodicity_breaking,
    fit_jump_distance_mixture,
    jump_distance_distribution,
)
from .tracking_attributes import (
    TRACKING_ATTRIBUTE_DESCRIPTIONS,
    TRACKING_ATTRIBUTE_NAMES,
    install_tracking_attributes,
    tracking_attribute_arrays,
)
from .tracking_export import export_tracking_result_zip
from .tracking_stats import (
    DatasetTrajectorySnapshot,
    PreparedTrajectories,
    TrackingMethodSpec,
    TrackingResult,
    get_tracking_method,
    kinematic_statistics,
    mean_squared_displacement,
    prepare_dataset_trajectories,
    prepare_trajectories,
    register_tracking_method,
    run_tracking_method,
    snapshot_dataset_trajectories,
    tracking_methods,
    trajectories_from_dataset,
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
    "export_tracking_result_zip",
    "TRACKING_ATTRIBUTE_DESCRIPTIONS",
    "TRACKING_ATTRIBUTE_NAMES",
    "install_tracking_attributes",
    "tracking_attribute_arrays",
    "TrackingStateAdapterSpec",
    "get_tracking_state_adapter",
    "register_tracking_state_adapter",
    "run_tracking_state_adapter",
    "tracking_state_adapter_availability",
    "tracking_state_adapters",
    "confinement_index",
    "ergodicity_breaking",
    "fit_jump_distance_mixture",
    "jump_distance_distribution",
    "PreparedTrajectories",
    "DatasetTrajectorySnapshot",
    "TrackingMethodSpec",
    "TrackingResult",
    "get_tracking_method",
    "kinematic_statistics",
    "mean_squared_displacement",
    "prepare_dataset_trajectories",
    "prepare_trajectories",
    "register_tracking_method",
    "run_tracking_method",
    "snapshot_dataset_trajectories",
    "tracking_methods",
    "trajectories_from_dataset",
]
