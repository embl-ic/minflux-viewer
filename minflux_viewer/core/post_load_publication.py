"""Qt-free publication of validated post-load proposals, without window access."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .dataset import MinfluxDataset


@dataclass(frozen=True)
class PostLoadChanges:
    calibration: bool
    attributes: bool
    logs: tuple[tuple[str, str], ...]


def publish_post_load(ds: MinfluxDataset, result: Mapping[str, Any]) -> PostLoadChanges:
    """Validate proposal shapes before writes; caller owns revision checks/notifications.

    Existing computations win over optional proposals. This is a synchronous
    dataset operation, not a worker callback or a rollback-on-allocation guarantee.
    """
    z = result.get("z_update") if "z_scaling_factor" not in ds.derived else None
    precision = result.get("loc_precision") if "sigma_per_trace_nm" not in ds.derived else None
    density = result.get("density") if "den" not in ds.attr else None
    last_density = result.get("last_density") if "den" not in ds.components.derived_last else None
    logs = tuple((str(message), str(level)) for message, level in result.get("logs", ()))
    z_value = None
    sizes = {}
    if z is not None:
        z_value = float(z["value"])
        if not np.isfinite(z_value) or z_value <= 0:
            raise ValueError("Post-load Z scaling factor must be positive and finite")
        source = str(z["source"])
        details = z.get("details")
        if details is not None:
            sizes = {f"z_scaling_factor_sizes_{axis}": details[axis].sizes for axis in "xyz"}
    if precision is not None:
        sigma = np.asarray(precision["per_trace_sigma_xyz"])
        ids = np.asarray(precision["trace_ids"])
        median = np.asarray(precision["median_sigma_xyz"], dtype=float)
        if ids.ndim != 1 or sigma.shape != (ids.size, 3) or median.shape != (3,):
            raise ValueError("Post-load precision must contain aligned trace IDs and XYZ sigma")
    if density is not None:
        density = np.asarray(density)
        if density.shape != (ds.prop.num_loc,):
            raise ValueError("Post-load density must align with materialized localizations")
    if last_density is not None:
        last_density = np.asarray(last_density)
        if last_density.ndim != 1:
            raise ValueError("Last-valid density must be one-dimensional")

    if z_value is not None:
        ds.set_z_scaling_factor(z_value, source=source)
        ds.derived["z_scaling_factor"] = np.asarray([z_value], dtype=float)
        for name, values in sizes.items():
            ds.derived[name] = values
    if precision is not None:
        for name, values in (("sigma_per_trace_nm", sigma), ("sigma_trace_ids", ids)):
            ds.attr[name] = values
            ds.derived[name] = values
        ds.cali.loc_precision = median
    if density is not None:
        for name in ("den", "local_density"):
            ds.attr[name] = density
            ds.derived[name] = density
            if name not in ds.prop.attr_names:
                ds.prop.attr_names.append(name)
    if last_density is not None:
        ds.components.derived_last["den"] = last_density
    return PostLoadChanges(z_value is not None, precision is not None or density is not None, logs)
