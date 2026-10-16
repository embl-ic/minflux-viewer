"""Prevent growth in broad exception catches and silently ignored failures.

Counts are per source file, not a claim that any existing catch is justified.
Run without arguments in CI; --update records decreases only. A code move must
transfer existing allowances explicitly, like the Ruff/mypy baseline.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tools" / "exception_baseline.json"
METRICS = ("broad", "silent")


def count_handlers(source: str) -> dict[str, int]:
    tree = ast.parse(source)
    names = {"Exception", "BaseException"}
    builtins = {"builtins"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "builtins":
            names.update(a.asname or a.name for a in node.names if a.name in names)
        elif isinstance(node, ast.Import):
            builtins.update(a.asname or a.name for a in node.names if a.name == "builtins")

    def broad(node):
        if node is None:
            return True
        if isinstance(node, ast.Name):
            return node.id in names
        if isinstance(node, ast.Attribute):
            return (isinstance(node.value, ast.Name) and node.value.id in builtins
                    and node.attr in {"Exception", "BaseException"})
        return isinstance(node, ast.Tuple) and any(broad(item) for item in node.elts)

    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler) and broad(n.type)]
    return {"broad": len(handlers),
            "silent": sum(len(n.body) == 1 and isinstance(n.body[0], ast.Pass) for n in handlers)}


def scan(package: Path) -> dict[str, dict[str, int]]:
    found = {}
    for path in sorted(package.rglob("*.py")):
        counts = count_handlers(path.read_text(encoding="utf-8-sig"))
        if counts["broad"]:
            found[path.relative_to(package.parent).as_posix()] = counts
    return found


def increases(baseline: dict, current: dict) -> list[tuple[str, str, int, int]]:
    return [(path, metric, baseline.get(path, {}).get(metric, 0), counts.get(metric, 0))
            for path, counts in sorted(current.items()) for metric in METRICS
            if counts.get(metric, 0) > baseline.get(path, {}).get(metric, 0)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--update", action="store_true", help="Record decreases only")
    mode.add_argument("--initialize", action="store_true", help="Create an absent baseline once")
    args = parser.parse_args(argv)
    current = scan(ROOT / "minflux_viewer")
    if args.initialize:
        if BASELINE.exists():
            parser.error("Baseline already exists; initialization cannot overwrite it")
        BASELINE.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    if not BASELINE.is_file():
        parser.error("Exception baseline is missing")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    added = increases(baseline, current)
    for path, metric, old, new in added:
        print(f"NEW {path} {metric}: {old} -> {new}")
    if args.update:
        # Refuse an increase rather than masking it while lowering other files.
        if added:
            print("Baseline not changed: fix the new catches first.")
            return 1
        BASELINE.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    totals = {metric: sum(c.get(metric, 0) for c in current.values()) for metric in METRICS}
    print(f"Broad exception handlers: {totals['broad']}; bare-pass handlers: {totals['silent']}")
    return int(bool(added))


if __name__ == "__main__":
    raise SystemExit(main())
