"""ROI in-region highlight gating by preferences (drawing view vs synced views)."""

import pytest

from minflux_viewer.core.roi_selection import roi_highlight_enabled


@pytest.mark.parametrize("in_roi,sync,is_source,expected", [
    (True,  True,  True,  True),    # default: drawing view highlights
    (True,  True,  False, True),    # default: other views highlight
    (False, True,  True,  False),   # counter-intuitive: drawing view OFF…
    (False, True,  False, True),    # …but synced views still ON
    (True,  False, True,  True),    # drawing view ON, others OFF
    (True,  False, False, False),
    (False, False, True,  False),   # both off
    (False, False, False, False),
])
def test_roi_highlight_enabled(in_roi, sync, is_source, expected):
    prefs = {"plot": {"roi_highlight_in_roi": in_roi, "roi_sync_highlight": sync}}
    assert roi_highlight_enabled(prefs, is_source=is_source) is expected


def test_defaults_enabled_when_keys_missing():
    assert roi_highlight_enabled({}, is_source=True) is True
    assert roi_highlight_enabled(None, is_source=False) is True
