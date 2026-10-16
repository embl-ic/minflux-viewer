"""Compatibility import for the relocated implementation."""
import sys

from ..utils import obf_compat as _implementation
from ..utils.obf_compat import *  # noqa: F403

sys.modules[__name__] = _implementation
