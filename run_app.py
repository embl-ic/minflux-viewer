"""
PyInstaller entry point.

PyInstaller executes the Analysis entry script as __main__, which breaks
relative imports inside minflux_viewer/__main__.py.  This thin wrapper
imports the package properly so all relative imports inside the package work.
"""
from minflux_viewer.__main__ import main

if __name__ == "__main__":
    main()
