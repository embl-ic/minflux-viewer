"""Compatibility import for the relocated implementation."""
import sys

from ..utils import zarr2 as _implementation
from ..utils.zarr2 import *  # noqa: F403

sys.modules[__name__] = _implementation
