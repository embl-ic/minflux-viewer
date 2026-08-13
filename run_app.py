"""
PyInstaller entry point.

PyInstaller executes the Analysis entry script as __main__, which breaks
relative imports inside minflux_viewer/__main__.py.  This thin wrapper
imports the package properly so all relative imports inside the package work.
"""
if __name__ == "__main__":
    import multiprocessing

    # PyInstaller replaces freeze_support() with a dispatcher for its frozen
    # multiprocessing helpers.  This must run before importing the application:
    # on macOS, numcodecs.blosc creates a multiprocessing lock when Zarr is first
    # imported, which starts a resource-tracker helper via this executable.  If
    # the helper is not diverted here, it runs the normal GUI entry point and
    # appears as another MINFLUX Viewer instance.
    multiprocessing.freeze_support()

    from minflux_viewer.__main__ import main

    raise SystemExit(main())
