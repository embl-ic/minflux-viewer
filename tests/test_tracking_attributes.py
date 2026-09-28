"""Materialized tracking attributes keep exclusions and provenance explicit."""

from __future__ import annotations

import numpy as np

from minflux_viewer.analysis.tracking_attributes import (
    install_tracking_attributes,
    tracking_attribute_arrays,
)
from minflux_viewer.analysis.tracking_stats import (
    kinematic_statistics,
    mean_squared_displacement,
    prepare_trajectories,
)


def _analysis():
    rng = np.random.default_rng(12)
    traces, per = 3, 14
    coords = np.cumsum(rng.normal(0.0, 4.0, (traces * per, 3)), axis=0)
    tid = np.repeat(np.arange(traces), per)
    tim = np.tile(np.arange(per) * 1e-3, traces)
    keep = np.ones(traces * per, dtype=bool)
    keep[4] = False
    prepared = prepare_trajectories(coords, tid, tim, keep=keep)
    msd = mean_squared_displacement(
        prepared, min_pairs=1, max_lag_points=4, min_segment_points=3)
    return prepared, msd, kinematic_statistics(prepared), keep


def test_tracking_attributes_map_back_to_source_rows_without_filling_gaps() -> None:
    prepared, msd, kinematics, keep = _analysis()
    arrays = tracking_attribute_arrays(
        keep.size, prepared, msd=msd, kinematics=kinematics)

    assert np.isnan(arrays["msd_d"][0][4])
    assert np.isnan(arrays["step_angle"][0][4])
    assert np.all(np.isnan(arrays["msd_d"][0][~keep]))
    assert arrays["msd_d"][1] == "nm²/s"
    assert arrays["msd_sigma_apparent"][2]["method_id"].endswith(".msd")


def test_install_makes_attributes_visible_and_auditable() -> None:
    from minflux_viewer.core.attributes import attribute_description, plot_attribute_names
    from minflux_viewer.core.dataset import build_localization_dataset

    prepared, msd, kinematics, _keep = _analysis()
    coords = np.zeros((prepared.diagnostics.n_input_rows, 3))
    ds = build_localization_dataset(
        name="derived",
        x_nm=coords[:, 0],
        y_nm=coords[:, 1],
        z_nm=coords[:, 2],
        tid=np.repeat(np.arange(3), 14),
        tim=np.tile(np.arange(14) * 1e-3, 3),
    )
    installed = install_tracking_attributes(
        ds, prepared, msd=msd, kinematics=kinematics)

    assert set(installed) == {
        "msd_d", "msd_alpha", "msd_sigma_apparent", "step_angle",
        "track_straightness",
    }
    assert ds.derived["msd_d"] is ds.attr["msd_d"]
    assert ds.mfx.meta["msd_d"]["user_visible"] is True
    assert "msd_d" in plot_attribute_names(ds, {"attributes": {"computed": []}})
    assert "tracking_derived_attributes" in ds.metadata
    assert "measured-lag MSD" in attribute_description("msd_d")
