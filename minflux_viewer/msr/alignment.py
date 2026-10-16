"""Compatibility import for the relocated implementation."""
import sys

from ..core import bead_alignment as _implementation
from ..core.bead_alignment import *  # noqa: F403

sys.modules[__name__] = _implementation
