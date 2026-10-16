"""Broad catches and silent broad catches may not grow per source file."""

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "tools" / "exception_ratchet.py"
_SPEC = importlib.util.spec_from_file_location("exception_ratchet", _PATH)
ratchet = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ratchet)


@pytest.mark.parametrize("clause", [
    "", "Exception", "BaseException", "(ValueError, Exception)",
    "builtins.Exception", "bi.BaseException", "Broad",
])
def test_counts_bare_named_qualified_tuple_and_import_alias_catches(clause):
    source = ("import builtins\nimport builtins as bi\n"
              "from builtins import Exception as Broad\n"
              f"try:\n    work()\nexcept{(' ' + clause) if clause else ''}:\n    pass\n")
    assert ratchet.count_handlers(source) == {"broad": 1, "silent": 1}


def test_counts_silent_separately_and_ignores_narrow_catches_comments_and_strings():
    source = '''
# except Exception: pass
text = "except: pass"
try:
    work()
except ValueError:
    pass
except Exception:
    log()
    raise
'''
    assert ratchet.count_handlers(source) == {"broad": 1, "silent": 0}


def test_new_file_or_silent_catch_cannot_borrow_another_files_allowance():
    baseline = {"a.py": {"broad": 2, "silent": 0}}
    current = {"a.py": {"broad": 1, "silent": 1}, "b.py": {"broad": 1, "silent": 0}}
    assert ratchet.increases(baseline, current) == [
        ("a.py", "silent", 0, 1), ("b.py", "broad", 0, 1),
    ]


def _workspace(tmp_path, monkeypatch):
    package = tmp_path / "minflux_viewer"
    package.mkdir()
    baseline = tmp_path / "baseline.json"
    monkeypatch.setattr(ratchet, "ROOT", tmp_path)
    monkeypatch.setattr(ratchet, "BASELINE", baseline)
    return package, baseline


def test_scan_uses_portable_relative_paths_and_omits_zero_counts(tmp_path):
    package = tmp_path / "package"
    nested = package / "nested"
    nested.mkdir(parents=True)
    (nested / "b.py").write_text("try:\n    work()\nexcept:\n    pass\n", encoding="utf-8")
    (package / "a.py").write_text("pass\n", encoding="utf-8")
    assert ratchet.scan(package) == {"package/nested/b.py": {"broad": 1, "silent": 1}}


def test_initialize_cannot_overwrite_existing_baseline(tmp_path, monkeypatch):
    _, baseline = _workspace(tmp_path, monkeypatch)
    assert ratchet.main(["--initialize"]) == 0
    before = baseline.read_bytes()
    with pytest.raises(SystemExit):
        ratchet.main(["--initialize"])
    assert baseline.read_bytes() == before


def test_update_refuses_increase_without_changing_any_allowances(tmp_path, monkeypatch):
    package, baseline = _workspace(tmp_path, monkeypatch)
    baseline.write_text('{"old.py": {"broad": 2, "silent": 1}}', encoding="utf-8")
    (package / "new.py").write_text("try:\n    work()\nexcept Exception:\n    pass\n", encoding="utf-8")
    before = baseline.read_bytes()
    assert ratchet.main([]) == 1
    assert ratchet.main(["--update"]) == 1
    assert baseline.read_bytes() == before


def test_update_locks_in_decreases_including_removed_files(tmp_path, monkeypatch):
    package, baseline = _workspace(tmp_path, monkeypatch)
    baseline.write_text(json.dumps({"minflux_viewer/a.py": {"broad": 2, "silent": 1},
                                    "minflux_viewer/gone.py": {"broad": 1, "silent": 1}}), encoding="utf-8")
    (package / "a.py").write_text("try:\n    work()\nexcept Exception:\n    raise\n", encoding="utf-8")
    assert ratchet.main(["--update"]) == 0
    assert json.loads(baseline.read_text(encoding="utf-8")) == {
        "minflux_viewer/a.py": {"broad": 1, "silent": 0},
    }


def test_current_package_does_not_exceed_exception_baseline():
    baseline = json.loads(ratchet.BASELINE.read_text(encoding="utf-8"))
    assert ratchet.increases(baseline, ratchet.scan(ratchet.ROOT / "minflux_viewer")) == []
