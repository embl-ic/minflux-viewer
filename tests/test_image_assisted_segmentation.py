from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from minflux_viewer.analysis import image_assisted_segmentation as assisted
from minflux_viewer.analysis.shape_segmentation import field_from_points
from minflux_viewer.core.confocal_mapping import (
    AssociatedImageCandidate,
    ConfocalCandidate,
)


def _candidate(field, raw_index, name):
    h, w = field.shape
    x0, y0 = field.origin_nm
    image = ConfocalCandidate(
        raw_index=raw_index,
        name=name,
        shape=(h, w),
        axes="YX",
        dtype="float64",
        x_start_m=x0 * 1e-9,
        y_start_m=y0 * 1e-9,
        x_step_m=field.pixel_nm * 1e-9,
        y_step_m=field.pixel_nm * 1e-9,
        z_start_m=None,
        z_step_m=None,
        bounds_xy_m=(
            (x0 * 1e-9, (x0 + w * field.pixel_nm) * 1e-9),
            (y0 * 1e-9, (y0 + h * field.pixel_nm) * 1e-9),
        ),
        matches=(),
    )
    return AssociatedImageCandidate(
        image=image,
        dataset_key="run",
        did="did-1",
        roi_bounds_xy_m=image.bounds_xy_m,
        coverage_area_ratio=1.0,
        generated=False,
    )


def test_associated_image_field_selects_by_agreement_and_cannot_invent_signal(
        monkeypatch, tmp_path):
    rng = np.random.default_rng(4)
    x = np.concatenate([rng.normal(500.0, 110.0, 900),
                        rng.normal(1500.0, 150.0, 1100)])
    y = np.concatenate([rng.normal(600.0, 180.0, 900),
                        rng.normal(1300.0, 140.0, 1100)])
    primary = field_from_points(x, y, 20.0)
    bad = _candidate(primary, 1, "unrelated")
    good = _candidate(primary, 2, "matching boundary")
    images = {
        1: rng.normal(size=primary.shape),
        2: ndi.gaussian_filter(primary.values, 5.0),
    }
    source = tmp_path / "source.msr"
    source.touch()
    monkeypatch.setattr(
        assisted, "discover_associated_image_candidates",
        lambda *_args, **_kwargs: [bad, good],
    )
    monkeypatch.setattr(
        assisted, "load_confocal_candidate_array",
        lambda _path, candidate: images[candidate.raw_index],
    )

    result = assisted.field_from_points_with_associated_image(
        x, y, msr_path=source, dataset_did="did-1",
        pixel_size_nm=20.0, max_source_pixel_nm=30.0,
    )

    assert result.provenance["used"] is True
    assert result.provenance["image_name"] == "matching boundary"
    assert result.provenance["generated_images_used"] is False
    assert result.provenance["agreement"] > 0.8
    assert np.all(result.field.values[primary.values == 0] == 0)
    assert np.all(result.field.values >= primary.values)


def test_associated_image_field_falls_back_when_source_identity_is_missing():
    x = np.array([0.0, 20.0, 40.0])
    y = np.array([0.0, 20.0, 40.0])
    result = assisted.field_from_points_with_associated_image(
        x, y, msr_path="missing.msr", dataset_did="",
    )
    assert result.provenance["used"] is False
    np.testing.assert_array_equal(
        result.field.values, field_from_points(x, y, 20.0).values)
