"""Compatibility import for the relocated implementation."""
import sys

from ..core import acquisition_metadata as _implementation
from ..core.acquisition_metadata import *  # noqa: F403

sys.modules[__name__] = _implementation
