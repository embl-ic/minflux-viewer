"""Command Finder metadata must point to shipped source owners."""
from pathlib import Path


def test_every_declared_command_source_exists():
    from minflux_viewer.ui.common.command_meta import COMMAND_META

    root = Path(__file__).resolve().parents[1]
    missing = {name: meta.source for name, meta in COMMAND_META.items()
               if meta.source and not (root / meta.source).is_file()}
    assert missing == {}
