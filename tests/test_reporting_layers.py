"""Numerical reporting does not load GUI or plugin execution infrastructure."""

import importlib
import subprocess
import sys

import pytest


def test_reporting_and_method_declarations_import_without_gui_or_plugin_loading():
    code = """
import sys
import minflux_viewer.analysis.precision_calculation
import minflux_viewer.analysis.precision_provenance
import minflux_viewer.analysis.method_text
import minflux_viewer.core.plugin_registry
import minflux_viewer.utils.plugin_method
assert not any(n.startswith(('PyQt6', 'pyqtgraph', 'minflux_viewer.ui', 'minflux_viewer.plugins')) for n in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("old,new", [
    ("analysis.localization_precision", "analysis.precision_calculation"),
    ("plugins.method", "utils.plugin_method"),
])
def test_compatibility_modules_keep_identity_and_monkeypatch_semantics(old, new):
    assert importlib.import_module("minflux_viewer." + old) is importlib.import_module("minflux_viewer." + new)


def test_plugin_registry_has_one_owner_and_keeps_the_legacy_interface():
    from minflux_viewer import plugins
    from minflux_viewer.core import plugin_registry

    for name in ("PluginEntry", "_REGISTRY", "available", "register"):
        assert getattr(plugins, name) is getattr(plugin_registry, name)


def test_report_rules_citations_and_provenance_have_single_owners():
    from minflux_viewer.analysis import (
        method_citations,
        method_rules,
        method_text,
        precision_calculation,
        precision_provenance,
    )

    assert method_text.RULES is method_rules.RULES
    assert method_text.CITE_STDDEV is method_citations.CITE_STDDEV
    assert precision_calculation.crlb_log_line is precision_provenance.crlb_log_line
