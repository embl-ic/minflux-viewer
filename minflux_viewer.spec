# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for MINFLUX Viewer — Windows one-directory build.
#
# Build:
#   .venv\Scripts\pyinstaller minflux_viewer.spec
#
# Output:  dist\minflux_viewer\minflux_viewer.exe

from pathlib import Path
import sys

ROOT = Path(SPECPATH)   # repo root (where this .spec lives)

# ---------------------------------------------------------------------------
# Data files to bundle
# ---------------------------------------------------------------------------
datas = [
    # Application resources (icons, UI files, sample filters)
    (str(ROOT / "resources"), "resources"),
]

# ---------------------------------------------------------------------------
# Hidden imports that PyInstaller's static analysis misses
# ---------------------------------------------------------------------------
hidden_imports = [
    # scipy — submodules loaded dynamically
    "scipy._lib.array_api_compat",
    "scipy._lib.array_api_compat.numpy",
    "scipy._lib.array_api_compat.numpy.fft",
    "scipy.ndimage._nd_image",
    "scipy.special._ufuncs_cxx",
    "scipy.linalg.cython_blas",
    "scipy.linalg.cython_lapack",
    # h5py
    "h5py._errors",
    "h5py.defs",
    "h5py.utils",
    "h5py.h5ac",
    "h5py._proxy",
    # matplotlib (used for colormaps only — agg + svg backends)
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_svg",
    # zarr
    "zarr",
    "zarr.storage",
    # tifffile
    "tifffile",
    # openpyxl
    "openpyxl",
    "openpyxl.cell._writer",
    # specpy (Abberior Windows SDK — present in this venv)
    "specpy",
    # roifile
    "roifile",
    # psutil Windows backend
    "psutil._pswindows",
    # minflux_viewer plugins (loaded at runtime via importlib)
    "minflux_viewer.plugins.msr_reader",
    "minflux_viewer.plugins.paraview",
    "minflux_viewer.plugins.generate_method_text",
]

# ---------------------------------------------------------------------------
# Modules to deliberately exclude (keeps the bundle smaller)
# ---------------------------------------------------------------------------
excludes = [
    # tkinter — Qt app, never needs Tk
    "tkinter",
    # Testing / dev tools — not needed at runtime
    "test",
    "tests",
    "pytest",
    "mypy",
    "ruff",
    # Heavy optional packages not used by this application
    "IPython",
    "notebook",
    "jupyter",
    "cv2",
    # NOTE: do NOT exclude stdlib modules (email, http, ipaddress, etc.).
    # Many are imported indirectly through the stdlib chain at boot time
    # (e.g. urllib.parse imports ipaddress; pathlib imports urllib.parse).
    # Excluding them causes ModuleNotFoundError in PyInstaller runtime hooks.
]

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    [str(ROOT / "minflux_viewer" / "__main__.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={
        "matplotlib": {
            "backends": ["agg", "svg"],
        },
    },
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=1,
)

# ---------------------------------------------------------------------------
# PYZ archive
# ---------------------------------------------------------------------------
pyz = PYZ(a.pure)

# ---------------------------------------------------------------------------
# EXE (one-directory mode — fast startup, easy to redistribute)
# ---------------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="minflux_viewer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # no console window (GUI app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "resources" / "icons" / "minflux_viewer_logo.png"),
)

# ---------------------------------------------------------------------------
# COLLECT — assemble the dist\minflux_viewer\ folder
# ---------------------------------------------------------------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="minflux_viewer",
)
