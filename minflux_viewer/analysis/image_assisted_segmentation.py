"""Calibrated acquisition-image support for point-cloud segmentation.

The point cloud remains the primary signal.  A spatially associated image is
selected without channel-name assumptions, resampled into the same physical
frame, and used as a deliberately weak multiplicative support field.  This
lets an independent acquisition image refine boundaries while ensuring that
image background cannot invent a new object where there are no localizations.

The module is project-neutral: callers provide only calibrated XY points, an
MSR source/DID, and resolution limits appropriate for their object size.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from ..core.confocal_mapping import (
    AssociatedImageCandidate,
    discover_associated_image_candidates,
    load_confocal_candidate_array,
)
from .shape_segmentation import ScalarField, field_from_points

__all__ = [
    "ImageAssistedFieldResult",
    "field_from_points_with_associated_image",
]


@dataclass(frozen=True)
class ImageAssistedFieldResult:
    """A segmentation field plus serializable image-selection provenance."""

    field: ScalarField
    provenance: dict


def _project_image(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 2:
        return np.asarray(values, dtype=float)
    if values.ndim == 3:
        # Accumulate in float64: integer channel stacks can overflow in their
        # source dtype even at modest Z depths.
        return np.sum(values, axis=0, dtype=np.float64)
    raise ValueError(f"Associated image must be YX or ZYX, got {values.shape}")


def _sample_on_field(image: np.ndarray, candidate: AssociatedImageCandidate,
                     field: ScalarField) -> np.ndarray:
    x_nm, y_nm = field.coords()
    calibrated = candidate.image
    col = ((x_nm - float(calibrated.x_start_m) * 1e9)
           / (float(calibrated.x_step_m) * 1e9) - 0.5)
    row = ((y_nm - float(calibrated.y_start_m) * 1e9)
           / (float(calibrated.y_step_m) * 1e9) - 0.5)
    return ndi.map_coordinates(
        np.asarray(image, dtype=float), [row, col], order=1,
        mode="constant", cval=np.nan, prefilter=False,
    )


def _normalise_support(values: np.ndarray, *, dark: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size < 16:
        return np.zeros_like(arr, dtype=float)
    lo, middle, hi = np.quantile(finite, [0.01, 0.50, 0.99])
    if dark:
        scale = float(middle - lo)
        signal = middle - arr
    else:
        scale = float(hi - middle)
        signal = arr - middle
    if not np.isfinite(scale) or scale <= 0.0:
        return np.zeros_like(arr, dtype=float)
    # The square root keeps a few very bright pixels from dominating the
    # agreement score or the weak support multiplier.
    support = np.sqrt(np.clip(signal / scale, 0.0, 1.0))
    return np.nan_to_num(support, nan=0.0, posinf=0.0, neginf=0.0)


def _agreement(primary: np.ndarray, support: np.ndarray) -> float:
    left = np.asarray(primary, dtype=float).ravel()
    right = np.asarray(support, dtype=float).ravel()
    finite = np.isfinite(left) & np.isfinite(right)
    if int(np.count_nonzero(finite)) < 16:
        return float("nan")
    left, right = left[finite], right[finite]
    left_sd, right_sd = float(left.std()), float(right.std())
    if left_sd <= 0.0 or right_sd <= 0.0:
        return float("nan")
    return float(np.mean(
        ((left - left.mean()) / left_sd)
        * ((right - right.mean()) / right_sd)
    ))


def _roi_bounds_nm(candidate: AssociatedImageCandidate):
    (x0, x1), (y0, y1) = candidate.roi_bounds_xy_m
    return (float(x0) * 1e9, float(y0) * 1e9,
            float(x1) * 1e9, float(y1) * 1e9)


def field_from_points_with_associated_image(
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    *,
    msr_path: str | Path,
    dataset_did: str,
    pixel_size_nm: float = 20.0,
    max_source_pixel_nm: float = 100.0,
    agreement_smoothing_nm: float = 100.0,
    min_agreement: float = 0.20,
    support_weight: float = 0.05,
) -> ImageAssistedFieldResult:
    """Build a point-density field with weak independent-image support.

    Every calibrated, non-generated image containing the dataset acquisition
    ROI is tried in both bright-object and dark-object polarity.  The channel
    whose image agrees best with the smoothed localization footprint is chosen.
    Images coarser than ``max_source_pixel_nm`` are ignored.  If discovery,
    reading, calibration, or agreement fails, the returned field is simply the
    localization density and ``provenance['used']`` is false.
    """
    primary = field_from_points(x_nm, y_nm, float(pixel_size_nm))
    base_provenance = {
        "used": False,
        "source_path": str(Path(msr_path)),
        "dataset_did": str(dataset_did or ""),
        "support_weight": float(support_weight),
        "selection": "calibrated containment + localization agreement",
        "generated_images_used": False,
        "candidates": [],
    }
    if not dataset_did or not Path(msr_path).is_file():
        return ImageAssistedFieldResult(primary, base_provenance)
    if not 0.0 <= float(support_weight) <= 1.0:
        raise ValueError("support_weight must lie in [0, 1]")

    candidates = discover_associated_image_candidates(
        msr_path,
        [{"dataset_key": str(dataset_did), "did": str(dataset_did)}],
        include_generated=False,
    )
    candidates = [
        item for item in candidates
        if max(item.pixel_xy_nm) <= float(max_source_pixel_nm)
    ]
    if not candidates:
        return ImageAssistedFieldResult(primary, base_provenance)

    smooth_sigma = max(float(agreement_smoothing_nm), 0.0) / primary.pixel_nm
    reference = (ndi.gaussian_filter(primary.values, smooth_sigma)
                 if smooth_sigma > 0 else np.asarray(primary.values, dtype=float))
    scored: list[tuple[float, float, float, AssociatedImageCandidate,
                       str, np.ndarray]] = []
    reports: list[dict] = []
    visible_bounds = _roi_bounds_nm(candidates[0])
    for candidate in candidates:
        try:
            image = _project_image(load_confocal_candidate_array(
                msr_path, candidate.image))
            sampled = _sample_on_field(image, candidate, primary)
        except Exception as exc:
            reports.append({
                "name": candidate.name,
                "raw_index": int(candidate.raw_index),
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        polarity_rows = []
        for dark, polarity in ((False, "bright"), (True, "dark")):
            support = _normalise_support(sampled, dark=dark)
            score = _agreement(reference, support)
            polarity_rows.append({"polarity": polarity, "agreement": score})
            if np.isfinite(score):
                # Agreement is primary.  Resolution and field specificity are
                # deterministic tie-breaks, never substitutes for agreement.
                scored.append((
                    float(score), -max(candidate.pixel_xy_nm),
                    -float(candidate.coverage_area_ratio), candidate,
                    polarity, support,
                ))
        reports.append({
            "name": candidate.name,
            "raw_index": int(candidate.raw_index),
            "pixel_xy_nm": [float(v) for v in candidate.pixel_xy_nm],
            "coverage_area_ratio": float(candidate.coverage_area_ratio),
            "polarities": polarity_rows,
        })

    provenance = {**base_provenance, "candidates": reports}
    if not scored:
        return ImageAssistedFieldResult(primary, provenance)
    score, _resolution, _coverage, candidate, polarity, support = max(
        scored, key=lambda row: row[:3])
    if score < float(min_agreement):
        provenance["best_agreement"] = float(score)
        provenance["reason"] = "no acquisition image agreed sufficiently with the localization footprint"
        return ImageAssistedFieldResult(primary, provenance)

    fused = np.asarray(primary.values, dtype=float) * (
        1.0 + float(support_weight) * np.asarray(support, dtype=float))
    field = ScalarField(
        fused, primary.pixel_nm, primary.origin_nm,
        visible_bounds_nm=visible_bounds,
    )
    provenance.update({
        "used": True,
        "image_name": candidate.name,
        "raw_index": int(candidate.raw_index),
        "polarity": polarity,
        "agreement": float(score),
        "pixel_xy_nm": [float(v) for v in candidate.pixel_xy_nm],
        "coverage_area_ratio": float(candidate.coverage_area_ratio),
    })
    return ImageAssistedFieldResult(field, provenance)
