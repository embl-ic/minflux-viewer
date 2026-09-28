"""Built-in HlyB/D pair-distance analysis plugin.

The workflow deliberately continues to use the public ``mfv`` facade even
though it ships with the application.  Keeping that boundary makes the
scientific workflow independently testable while registering it like every
other built-in plugin, so frozen builds collect it automatically.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from .. import PluginEntry, register
from ..manifest import read_manifest

_MANIFEST = read_manifest(Path(__file__).resolve().parent)


def _launch(state, parent=None):
    """Run the API-based workflow with normal plugin attribution."""
    from .main import run

    ctx = state.mfv
    calls = getattr(ctx, "calls", None)
    suppression = calls.suppress() if calls is not None else contextlib.nullcontext()
    scope = getattr(ctx, "plugin_scope", None)
    running = (
        scope(_MANIFEST.id, _MANIFEST.label, _MANIFEST.method)
        if scope is not None
        else contextlib.nullcontext()
    )
    with suppression, running:
        return run(ctx)


register(PluginEntry(
    name=_MANIFEST.label,
    tooltip=_MANIFEST.description,
    launch=_launch,
    keywords=_MANIFEST.keywords,
    source=str(_MANIFEST.entry_path),
    plugin_id=_MANIFEST.id,
    method=_MANIFEST.method,
))
