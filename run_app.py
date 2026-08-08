"""
PyInstaller entry point.

PyInstaller executes the Analysis entry script as __main__, which breaks
relative imports inside minflux_viewer/__main__.py.  This thin wrapper
imports the package properly so all relative imports inside the package work.
"""
import sys

from minflux_viewer.__main__ import main

if __name__ == "__main__":
    # Propagate the exit code: main() returns 0 without building a UI when it
    # handed macOS startup documents to a running viewer
    # (ui/document_open_relay.py), and a nonzero code on failure. Calling
    # main() bare discarded both.
    sys.exit(main())
