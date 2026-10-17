"""Pure publication validates proposals before touching scientific state."""

import numpy as np
import pytest

from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.core.post_load_publication import publish_post_load


def _dataset():
    return build_localization_dataset(name="example", x_nm=np.arange(5.), y_nm=np.arange(5.))


@pytest.mark.parametrize("shape", [(), (1,), (4,), (6,), (5, 1), (1, 5), (2, 3)])
def test_bad_density_cannot_publish_an_earlier_calibration_proposal(shape):
    ds = _dataset()
    previous = ds.cali.z_scaling_factor
    with pytest.raises(ValueError, match="density"):
        publish_post_load(ds, {"z_update": {"value": .75, "source": "test"},
                               "density": np.zeros(shape)})
    assert ds.cali.z_scaling_factor == previous
    assert "z_scaling_factor" not in ds.derived


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -.1])
def test_invalid_calibration_cannot_be_published(value):
    with pytest.raises(ValueError, match="positive and finite"):
        publish_post_load(_dataset(), {"z_update": {"value": value, "source": "test"}})


def test_valid_publication_reports_only_changed_aspects():
    ds = _dataset()
    density = np.arange(5.)
    changes = publish_post_load(ds, {"density": density, "logs": [("computed", "INFO")]})
    assert changes.attributes and not changes.calibration
    assert changes.logs == (("computed", "INFO"),)
    assert ds.attr["den"] is ds.derived["local_density"]
    again = publish_post_load(ds, {"density": np.zeros(2)})
    assert not again.attributes  # Existing result wins; discarded proposal is unused.


@pytest.mark.parametrize("shape", [(), (1,), (4,), (6,), (5, 1), (1, 5), (2, 3)])
def test_malformed_imported_masks_are_not_flattened_or_silently_dropped(shape):
    from minflux_viewer.core.loader_metadata_recipe import _restore_filter_mask

    ds = _dataset()
    previous = ds.filter_mask.copy()
    with pytest.raises(ValueError, match="align"):
        _restore_filter_mask(ds, np.ones(shape))
    np.testing.assert_array_equal(ds.filter_mask, previous)
