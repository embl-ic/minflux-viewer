import numpy as np

from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.core.roi import RoiRecord
from minflux_viewer.core.roi_selection import (
    ROI_MASKS_STATE_KEY,
    rectangle_mask,
    store_roi_mask,
    value_range_mask,
)


def test_rectangle_mask_respects_bounds_and_base_mask() -> None:
    record = RoiRecord.create("rectangle", {"bounds": [0.0, 0.0, 10.0, 5.0]})
    x = np.array([-1.0, 0.0, 5.0, 10.0, 11.0])
    y = np.array([2.0, 0.0, 6.0, 5.0, 3.0])
    base = np.array([True, False, True, True, True])

    mask = rectangle_mask(x, y, record, base_mask=base)

    assert mask.tolist() == [False, False, False, True, False]


def test_value_range_mask_uses_inclusive_limits() -> None:
    values = np.array([0.0, 1.0, 2.0, np.nan, 3.0])

    mask = value_range_mask(values, 1.0, 2.0)

    assert mask.tolist() == [False, True, True, False, False]


def test_store_roi_mask_caches_derived_mask_and_metadata() -> None:
    ds = build_localization_dataset(
        name="roi-data",
        x_nm=np.array([0.0, 1.0, 2.0]),
        y_nm=np.array([0.0, 1.0, 2.0]),
    )
    record = RoiRecord.create("rectangle", {"bounds": [0.0, 0.0, 1.5, 1.5]})
    mask = np.array([True, True, False])

    key = store_roi_mask(ds, record, mask, context={"source_view": "test"})

    assert key == record.mask_key
    assert ds.derived[key].tolist() == [True, True, False]
    assert ds.state[ROI_MASKS_STATE_KEY][record.id]["selected_count"] == 2
    assert record.selected_count == 2
    assert record.selection_dirty is False
