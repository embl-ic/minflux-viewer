"""The API's own descriptions follow the API.

``__api_version__`` went to 1.3 when ``mfv.plugins`` was added, while the
Script Editor's *API Help* and docs/scripting-api.md went on saying 1.2 and
listing nine namespaces. Nothing failed, so nothing noticed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minflux_viewer.api import NAMESPACES, __api_version__

REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "scripting-api.md"


def test_the_editor_help_names_the_version_and_every_namespace():
    pytest.importorskip("PyQt6")
    from minflux_viewer.ui.tools.script_editor_window import _API_HELP, _COMPLETION_MAP

    assert _API_HELP.startswith(f"MINFLUX Viewer scripting API {__api_version__}\n")
    for name in NAMESPACES:
        assert f"mfv.{name} " in _API_HELP, name
        assert name in _COMPLETION_MAP["mfv"], name
        assert f"mfv.{name}" in _COMPLETION_MAP, name


def test_the_reference_names_the_version_and_every_namespace():
    text = REFERENCE.read_text(encoding="utf-8")
    assert f"**API version {__api_version__}.**" in text
    for name in NAMESPACES:
        assert f"| `mfv.{name}` |" in text, name
        assert f"## `mfv.{name}`" in text, name
