"""Hold the Ruff and mypy findings to a baseline that only ever goes down.

The code base carries existing findings (about 250 from Ruff and 1,570 from
mypy when this was introduced) that cannot all be fixed at once. Rather than
ignore them, ``tools/lint_baseline.json`` records how many there are of each
rule in each file, and this check fails when any of those counts RISES -- so
no change can add a finding, while every fix is free to land.

Counts are kept per (file, rule), not per line, so editing a file does not
shift its findings out of the baseline.

    python tools/lint_ratchet.py               # check (what CI runs)
    python tools/lint_ratchet.py --update      # lock in fixes: lower the baseline
    python tools/lint_ratchet.py --update --accept-increase   # deliberate, rare

``--update`` only ever lowers counts. Raising one needs ``--accept-increase``,
so a new finding is always a decision someone made, never a side effect.

mypy evaluates ``sys.platform`` branches as the machine it runs on, so the
baseline is measured on Windows and CI runs this check on Windows.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tools" / "lint_baseline.json"
TOOLS = ("ruff", "mypy")

_MYPY_LINE = re.compile(
    r"^(?P<path>[^:]+(?::\\[^:]+)?\.pyi?):\d+(?::\d+)?: error: .*?"
    r"(?:\s+\[(?P<code>[a-z0-9-]+)\])?\s*$")

Counts = dict[str, dict[str, int]]      # {path: {rule: count}}


def _relative(path: str) -> str:
    p = Path(path)
    try:
        p = p.resolve().relative_to(ROOT)
    except (OSError, ValueError):
        pass
    return p.as_posix()


def _add(counts: Counts, path: str, code: str) -> None:
    rules = counts.setdefault(_relative(path), {})
    rules[code] = rules.get(code, 0) + 1


def parse_ruff(json_text: str) -> Counts:
    counts: Counts = {}
    for finding in json.loads(json_text or "[]"):
        _add(counts, finding["filename"], finding.get("code") or "syntax-error")
    return counts


def parse_mypy(text: str) -> Counts:
    counts: Counts = {}
    for line in text.splitlines():
        match = _MYPY_LINE.match(line.strip())
        if match:
            _add(counts, match["path"], match["code"] or "misc")
    return counts


def run_ruff() -> Counts:
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", ".", "--output-format", "json",
         "--exit-zero"], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ruff could not run:\n{result.stderr}")
    return parse_ruff(result.stdout)


def run_mypy() -> Counts:
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "minflux_viewer"],
        cwd=ROOT, capture_output=True, text=True)
    if result.returncode not in (0, 1):     # 1 = findings; 2 = mypy itself failed
        raise SystemExit(f"mypy could not run:\n{result.stdout}{result.stderr}")
    return parse_mypy(result.stdout)


def compare(baseline: Counts, current: Counts) -> tuple[list, list]:
    """``(increases, decreases)`` as ``(path, rule, baseline, now)`` rows."""
    up, down = [], []
    for path in sorted(set(baseline) | set(current)):
        before, now = baseline.get(path, {}), current.get(path, {})
        for rule in sorted(set(before) | set(now)):
            b, n = before.get(rule, 0), now.get(rule, 0)
            if n > b:
                up.append((path, rule, b, n))
            elif n < b:
                down.append((path, rule, b, n))
    return up, down


def ratcheted(baseline: Counts, current: Counts, *, accept_increase: bool) -> Counts:
    """The new baseline: each count lowered to what it is now, never raised
    unless ``accept_increase``. Zero counts and empty files are dropped."""
    out: Counts = {}
    for path in set(baseline) | set(current):
        before, now = baseline.get(path, {}), current.get(path, {})
        for rule in set(before) | set(now):
            b, n = before.get(rule, 0), now.get(rule, 0)
            value = n if (accept_increase or n <= b) else b
            if value:
                out.setdefault(path, {})[rule] = value
    return {p: dict(sorted(r.items())) for p, r in sorted(out.items())}


def _total(counts: Counts) -> int:
    return sum(sum(rules.values()) for rules in counts.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--update", action="store_true",
                        help="lower the baseline to the current counts")
    parser.add_argument("--accept-increase", action="store_true",
                        help="with --update, also raise counts that went up")
    parser.add_argument("--only", choices=TOOLS, action="append",
                        help="check one tool (repeatable); default: both")
    args = parser.parse_args(argv)

    stored = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
    failed = False
    for tool in args.only or TOOLS:
        current = run_ruff() if tool == "ruff" else run_mypy()
        baseline = stored.get(tool, {})
        up, down = compare(baseline, current)
        print(f"{tool}: {_total(current)} finding(s), baseline {_total(baseline)}")
        for path, rule, b, n in up:
            print(f"  NEW   {path}  {rule}: {b} -> {n}")
        if down:
            print(f"  {sum(b - n for *_, b, n in down)} fewer than the baseline in "
                  f"{len({row[0] for row in down})} file(s); run with --update to "
                  "lock that in.")
        if args.update:
            stored[tool] = ratcheted(baseline, current,
                                     accept_increase=args.accept_increase)
        if up and not (args.update and args.accept_increase):
            failed = True
    if args.update:
        BASELINE.write_text(json.dumps(stored, indent=1, sort_keys=True) + "\n",
                            encoding="utf-8")
        print(f"Wrote {BASELINE.relative_to(ROOT).as_posix()}.")
    if failed:
        print("\nA change added Ruff or mypy findings (NEW above). Fix them; an "
              "increase that is genuinely intended needs "
              "`--update --accept-increase`, committed with the reason.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
