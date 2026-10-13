"""tools/lint_ratchet.py: findings may go down, never up."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "tools" / "lint_ratchet.py"
_spec = importlib.util.spec_from_file_location("lint_ratchet", _PATH)
ratchet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ratchet)


def test_mypy_findings_are_counted_per_file_and_rule_whatever_the_separator():
    text = "\n".join([
        r"minflux_viewer\ui\a.py:12: error: Incompatible types  [assignment]",
        "minflux_viewer/ui/a.py:40:5: error: Item None has no attribute  [union-attr]",
        r"minflux_viewer\ui\a.py:41: error: Incompatible types  [assignment]",
        r"minflux_viewer\core\b.py:3: error: A message with no code",
        r"minflux_viewer\core\b.py:3: note: See https://...",
        "Found 4 errors in 2 files (checked 2 source files)",
    ])
    assert ratchet.parse_mypy(text) == {
        "minflux_viewer/ui/a.py": {"assignment": 2, "union-attr": 1},
        "minflux_viewer/core/b.py": {"misc": 1},
    }


def test_ruff_findings_are_counted_per_file_and_rule():
    root = ratchet.ROOT
    payload = json.dumps([
        {"filename": str(root / "tools" / "x.py"), "code": "F401"},
        {"filename": str(root / "tools" / "x.py"), "code": "F401"},
        {"filename": str(root / "tools" / "x.py"), "code": None},
    ])
    assert ratchet.parse_ruff(payload) == {
        "tools/x.py": {"F401": 2, "syntax-error": 1}}


def test_only_a_rise_is_reported_as_new():
    baseline = {"a.py": {"F401": 2, "E741": 1}}
    current = {"a.py": {"F401": 3}, "b.py": {"UP031": 1}}
    up, down = ratchet.compare(baseline, current)
    assert up == [("a.py", "F401", 2, 3), ("b.py", "UP031", 0, 1)]
    assert down == [("a.py", "E741", 1, 0)]


def test_update_lowers_but_never_raises_without_being_told():
    baseline = {"a.py": {"F401": 2, "E741": 1}}
    current = {"a.py": {"F401": 3}, "b.py": {"UP031": 1}}
    assert ratchet.ratcheted(baseline, current, accept_increase=False) == {
        "a.py": {"F401": 2}}                       # E741 fixed -> gone; rises kept out
    assert ratchet.ratcheted(baseline, current, accept_increase=True) == {
        "a.py": {"F401": 3}, "b.py": {"UP031": 1}}
