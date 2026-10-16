"""Composition of numerical steps used by the post-load worker."""
from ..core.post_load import PostLoadAlgorithms
from .local_density import compute_local_density_for_points
from .precision_calculation import stddev_per_trace
from .trace_analysis import estimate_anisotropy_for_dataset

POST_LOAD_ALGORITHMS = PostLoadAlgorithms(
    estimate_anisotropy=estimate_anisotropy_for_dataset,
    trace_precision=stddev_per_trace,
    density=compute_local_density_for_points,
)
