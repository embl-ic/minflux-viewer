"""Broad catches and silent broad catches may not grow per source file."""

import ast
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "tools" / "exception_ratchet.py"
_SPEC = importlib.util.spec_from_file_location("exception_ratchet", _PATH)
ratchet = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ratchet)


# These are terminal callbacks, decoder boundaries or cleanup-and-reraise,
# never permission to substitute different scientific data on execution error.
_BACKEND_BOUNDARIES = {
    "core/app_state.py:AppState.add_dataset": (3, "Optional reporting after dataset admission"),
    "core/app_state.py:AppState.log": (1, "Attribution failure reported on stderr"),
    "core/app_state.py:AppState.task": (1, "Mark failed and re-raise"),
    "core/coordinates.py:CoordinateService._get": (1, "Propagate builder error to waiting callers and re-raise"),
    "core/minflux_zarr.py:_replace_viewer_directories": (1, "Rollback and re-raise"),
    "core/minflux_zarr.py:write_minflux_zarr": (1, "Restore backup and re-raise"),
    "core/minflux_zarr.py:write_minflux_zarr_project": (1, "Restore backup and re-raise"),
    "core/task_registry.py:TaskRegistry.request_stop": (1, "Report arbitrary cancellation callback failure"),
    "core/updater.py:check_for_update": (1, "Explicit error result at network boundary"),
    "core/user_libs.py:discard_install": (1, "Report cancellation cleanup failure"),
    "analysis/method_text.py:_collect": (1, "Report broken plugin declaration, retain event; do not guess attribution"),
    "msr/legacy_mbm.py:build_legacy_mbm": (1, "Report caller-supplied logging failure"),
    "msr/msr_parser.py:_ome_xml_metadata": (1, "Opaque optional decoder; propagate allocation failure"),
    "msr/msr_parser.py:LegacyMSRParser.build_series_tree_entries": (2, "Opaque decoder and warning callback; report failures"),
    "msr/msr_parser.py:GeneralMSRParser.parse": (1, "Report opaque decoder error; propagate allocation failure"),
    "msr/msr_parser.py:GeneralMSRParser._mfxdta_dataset_entry": (2, "Report malformed stack before publication; propagate allocation failure"),
    "msr/session_export.py:export_image_series": (1, "Report failed export and re-raise; always close source"),
}


def test_backend_broad_boundaries_are_reviewed_per_function_not_just_per_file():
    package = _PATH.parents[1] / "minflux_viewer"
    found = Counter()
    for layer in ("core", "analysis", "msr"):
        for path in (package / layer).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
            for handler in (node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)):
                # Reuse the ratchet's bare/qualified/tuple/import-alias semantics.
                imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
                probe = ast.Try(body=[ast.Pass()], handlers=[ast.ExceptHandler(
                    type=handler.type, name=None, body=[ast.Pass()])], orelse=[], finalbody=[])
                if not ratchet.count_handlers(ast.unparse(ast.Module(body=[*imports, probe], type_ignores=[])))["broad"]:
                    continue
                owner, names = handler, []
                while owner in parents:
                    owner = parents[owner]
                    if isinstance(owner, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        names.append(owner.name)
                key = f"{path.relative_to(package).as_posix()}:{'.'.join(reversed(names))}"
                found[key] += 1
                assert not (len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass)), key
    assert dict(found) == {key: count for key, (count, reason) in _BACKEND_BOUNDARIES.items()}


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
