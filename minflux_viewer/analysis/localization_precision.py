"""Compatibility alias for localization-precision calculations and UI exports."""
import sys

from . import precision_calculation as _implementation
from .precision_calculation import BG_CENTRE as BG_CENTRE
from .precision_calculation import BG_CENTRE_EARLIER as BG_CENTRE_EARLIER
from .precision_calculation import BG_FBG as BG_FBG
from .precision_calculation import BG_NONE as BG_NONE
from .precision_calculation import CRLB_DEFAULT_L_NM as CRLB_DEFAULT_L_NM
from .precision_calculation import CRLB_DEFAULT_LZ_NM as CRLB_DEFAULT_LZ_NM
from .precision_calculation import CRLB_DEFAULT_SIGMA_Q_NM as CRLB_DEFAULT_SIGMA_Q_NM
from .precision_calculation import FRC_DEFAULT_IMAGE_PX as FRC_DEFAULT_IMAGE_PX
from .precision_calculation import FRC_DEFAULT_REPEATS as FRC_DEFAULT_REPEATS
from .precision_calculation import FRC_MAX_IMAGE_PX as FRC_MAX_IMAGE_PX
from .precision_calculation import FRC_TARGET_PIXEL_NM as FRC_TARGET_PIXEL_NM
from .precision_calculation import FRC_THRESHOLD as FRC_THRESHOLD
from .precision_calculation import MIN_LOCS_PER_TRACE as MIN_LOCS_PER_TRACE
from .precision_calculation import CRLBResult as CRLBResult
from .precision_calculation import FRCResult as FRCResult
from .precision_calculation import crlb_from_inputs as crlb_from_inputs
from .precision_calculation import crlb_inputs_for_dataset as crlb_inputs_for_dataset
from .precision_calculation import crlb_inputs_from_arrays as crlb_inputs_from_arrays
from .precision_calculation import crlb_log_line as crlb_log_line
from .precision_calculation import crlb_matched_comparison as crlb_matched_comparison
from .precision_calculation import crlb_precision as crlb_precision
from .precision_calculation import excess_precision as excess_precision
from .precision_calculation import frc_resolution as frc_resolution
from .precision_calculation import stddev_per_trace as stddev_per_trace


def __getattr__(name):
    return getattr(_implementation, name)


sys.modules[__name__] = _implementation
