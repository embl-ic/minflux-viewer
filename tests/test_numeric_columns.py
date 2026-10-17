"""Exact numeric buffering boundaries and explicit precision-risk flags."""

import numpy as np
import pytest

from minflux_viewer.core.numeric_columns import NumericColumnBuffer


@pytest.mark.parametrize("value,kind", [(2**53 + 1, "i"), (2**63 - 1, "i"),
                                       (-(2**63), "i"), (2**64 - 1, "u")])
def test_full_width_integer_text_is_exact(value, kind):
    buffer = NumericColumnBuffer()
    buffer.append(str(value))
    assert buffer.values().dtype.kind == kind
    assert int(buffer.values()[0]) == value
    assert not buffer.precision_loss


@pytest.mark.parametrize("cells", [
    [str(2**53 + 1), ""], ["", str(2**53 + 1)],
    [str(2**53 + 1), "1.5"], ["1.5", str(2**53 + 1)],
    [str(2**64 - 1), "-1"], ["-1", str(2**64 - 1)],
    [str(2**65)], [str(-(2**64))], [float(2**53 + 1)],
])
def test_inexact_or_unrepresentable_integer_columns_are_flagged(cells):
    buffer = NumericColumnBuffer()
    for cell in cells:
        buffer.append(cell)
    assert buffer.precision_loss


def test_missing_values_and_floats_keep_normal_numeric_semantics():
    buffer = NumericColumnBuffer()
    for cell in ("1.25", None, "", "invalid", "2"):
        buffer.append(cell)
    np.testing.assert_array_equal(buffer.values(), [1.25, np.nan, np.nan, np.nan, 2])
    assert not buffer.precision_loss
