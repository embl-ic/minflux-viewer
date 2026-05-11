#!/usr/bin/env python3
"""
tools/compile_ui.py
====================
Compile Qt Designer ``.ui`` files in ``resources/ui/`` into Python modules.

Run from the project root::

    poetry run python tools/compile_ui.py

Output
------
For each ``resources/ui/foo.ui`` this produces
``minflux_viewer/ui/generated/foo_ui.py`` with a ``Ui_Foo`` class that can
be used either:

* By subclassing: ``class FooWindow(QMainWindow, Ui_Foo): self.setupUi(self)``
* Or by calling ``Ui_Foo().setupUi(target_widget)`` on an existing widget.

Options
-------
* ``--watch``  Re-run whenever a ``.ui`` file changes (requires ``watchdog``).
* ``--check``  Only verify that the generated Python is up to date; exit non-zero
               otherwise. Useful for CI / pre-commit hooks.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
UI_SRC_DIR   = PROJECT_ROOT / "resources" / "ui"
UI_OUT_DIR   = PROJECT_ROOT / "minflux_viewer" / "ui" / "generated"

_HEADER = (
    "# -*- coding: utf-8 -*-\n"
    "# AUTO-GENERATED FROM .ui — DO NOT EDIT BY HAND.\n"
    "# Re-run: poetry run python tools/compile_ui.py\n"
)


def compile_one(ui_file: Path, out_file: Path) -> None:
    """Compile a single .ui file to Python via pyuic6."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "pyuic6",
        str(ui_file.relative_to(PROJECT_ROOT)),
        "-o",
        str(out_file.relative_to(PROJECT_ROOT)),
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    # Prepend our own header so diffs stay readable
    body = out_file.read_text(encoding="utf-8")
    out_file.write_text(_HEADER + body, encoding="utf-8")
    print(f"  {ui_file.relative_to(PROJECT_ROOT)}  ->  {out_file.relative_to(PROJECT_ROOT)}")


def all_sources() -> list[Path]:
    return sorted(UI_SRC_DIR.glob("*.ui"))


def target_for(ui_file: Path) -> Path:
    stem = ui_file.stem
    return UI_OUT_DIR / f"{stem}_ui.py"


def ensure_init_py() -> None:
    init = UI_OUT_DIR / "__init__.py"
    init.parent.mkdir(parents=True, exist_ok=True)
    if not init.exists():
        init.write_text(
            '"""Auto-generated UI modules from resources/ui/*.ui.\n\n'
            "Do not edit by hand — regenerate with ``tools/compile_ui.py``.\n"
            '"""\n',
            encoding="utf-8",
        )


def cmd_build() -> int:
    sources = all_sources()
    if not sources:
        print(f"No .ui files found in {UI_SRC_DIR}")
        return 0

    ensure_init_py()
    print(f"Compiling {len(sources)} .ui file(s) from {UI_SRC_DIR}:")
    for ui in sources:
        compile_one(ui, target_for(ui))
    return 0


def cmd_check() -> int:
    """Verify that every .ui has an up-to-date generated module."""
    sources  = all_sources()
    outdated = []
    for ui in sources:
        out = target_for(ui)
        if not out.exists() or out.stat().st_mtime < ui.stat().st_mtime:
            outdated.append(ui.name)
    if outdated:
        print("Outdated generated modules for:", outdated)
        print("Run: poetry run python tools/compile_ui.py")
        return 1
    print("All generated modules up to date.")
    return 0


def cmd_watch() -> int:
    try:
        from watchdog.events   import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        print("The --watch mode requires the 'watchdog' package.")
        print("Install it with:  pip install watchdog")
        return 2

    import time

    class Handler(FileSystemEventHandler):
        def on_modified(self, event):
            p = Path(event.src_path)
            if p.suffix == ".ui":
                print(f"Changed: {p.name}")
                try:
                    compile_one(p, target_for(p))
                except subprocess.CalledProcessError as e:
                    print(f"  pyuic6 failed: {e}")

    cmd_build()
    observer = Observer()
    observer.schedule(Handler(), str(UI_SRC_DIR), recursive=False)
    observer.start()
    print(f"Watching {UI_SRC_DIR} — Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Compile .ui files to Python.")
    ap.add_argument("--check", action="store_true",
                    help="Verify generated modules are up to date; exit non-zero otherwise.")
    ap.add_argument("--watch", action="store_true",
                    help="Continuously rebuild when a .ui file changes.")
    args = ap.parse_args()

    if args.check:
        return cmd_check()
    if args.watch:
        return cmd_watch()
    return cmd_build()


if __name__ == "__main__":
    sys.exit(main())
