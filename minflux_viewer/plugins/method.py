"""Compatibility alias for the Qt-free plugin method declarations."""
import sys

from ..utils import plugin_method as _implementation
from ..utils.plugin_method import MISSING_TEMPLATE as MISSING_TEMPLATE
from ..utils.plugin_method import SECTIONS as SECTIONS
from ..utils.plugin_method import MethodError as MethodError
from ..utils.plugin_method import MethodField as MethodField
from ..utils.plugin_method import MethodSpec as MethodSpec
from ..utils.plugin_method import format_value as format_value
from ..utils.plugin_method import parse_method_spec as parse_method_spec
from ..utils.plugin_method import render_method_text as render_method_text
from ..utils.plugin_method import substitute as substitute

sys.modules[__name__] = _implementation
