# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for MINFLUX Viewer — Windows one-directory build.
#
# Build:
#   .venv\Scripts\pyinstaller minflux_viewer.spec
#
# Output:  dist\minflux_viewer\minflux_viewer.exe

from pathlib import Path
import re
import sys

ROOT = Path(SPECPATH)   # repo root (where this .spec lives)

# App version, parsed from the package without importing it.
VERSION = re.search(
    r'__version__\s*=\s*"([^"]+)"',
    (ROOT / "minflux_viewer" / "__init__.py").read_text(encoding="utf-8"),
).group(1)

# psutil ships a per-OS backend; name the right one so static analysis finds it.
PSUTIL_BACKEND = {"win32": "psutil._pswindows",
                  "darwin": "psutil._psosx"}.get(sys.platform, "psutil._pslinux")

# Icons: a .png works for the Windows exe; a macOS .app wants .icns (optional).
EXE_ICON = str(ROOT / "resources" / "icons" / "minflux_viewer_logo.png") \
    if sys.platform == "win32" else None
_ICNS = ROOT / "resources" / "icons" / "minflux_viewer_logo.icns"
MAC_ICON = str(_ICNS) if _ICNS.exists() else None

# UPX shrinks the Windows build, but on macOS it corrupts Mach-O binaries — a
# UPX-compressed libpython fails to load at boot with
# "PYI-xxxxx:ERROR: failed to load python shared library" — so disable it there.
USE_UPX = sys.platform != "darwin"

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
    # msr-reader (pure-Python OBF/.msr reader)
    "msr_reader",
    "msr_reader.obffile",
    # roifile
    "roifile",
    # psutil per-OS backend (Windows/macOS/Linux)
    PSUTIL_BACKEND,
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
    [str(ROOT / "run_app.py")],
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
    upx=USE_UPX,
    console=False,          # no console window (GUI app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=EXE_ICON,
)

# ---------------------------------------------------------------------------
# COLLECT — assemble the dist\minflux_viewer\ folder
# ---------------------------------------------------------------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=USE_UPX,
    upx_exclude=[],
    name="minflux_viewer",
)

# ---------------------------------------------------------------------------
# macOS — wrap the one-folder build into a clickable .app bundle
# (Windows/Linux ship the COLLECT folder as-is.)
# ---------------------------------------------------------------------------
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="MINFLUX Viewer.app",
        icon=MAC_ICON,
        bundle_identifier="eu.embl.ic.minflux-viewer",
        version=VERSION,
        info_plist={
            "CFBundleName": "MINFLUX Viewer",
            "CFBundleDisplayName": "MINFLUX Viewer",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
            # Declare the documents we open.  Without this, Launch Services has
            # no registered handler for e.g. ".msr", so dropping one on the app
            # icon does NOT reach the running instance — macOS launches a second
            # copy instead.  The running instance receives the document as a
            # QFileOpenEvent, handled by ui/file_open_app.py; the two changes
            # are required together, neither works on its own.
            "CFBundleDocumentTypes": [
                {
                    "CFBundleTypeName": "MINFLUX data",
                    "CFBundleTypeRole": "Viewer",
                    # "Alternate", not "Owner": Imspector owns .msr on a machine
                    # that has it, and we should not take the association over.
                    "LSHandlerRank": "Alternate",
                    "LSTypeIsPackage": False,
                    "CFBundleTypeExtensions": [
                        "msr", "mat", "npy", "json",
                        "csv", "tsv", "txt", "xlsx", "xlsm",
                        "tif", "tiff", "roi",
                    ],
                },
                {
                    # A Zarr store is a *directory*, so it needs its own entry
                    # with LSTypeIsPackage — otherwise Finder treats it as a
                    # folder to open rather than a document to hand over.
                    "CFBundleTypeName": "MINFLUX Zarr store",
                    "CFBundleTypeRole": "Viewer",
                    "LSHandlerRank": "Alternate",
                    "LSTypeIsPackage": True,
                    "CFBundleTypeExtensions": ["zarr"],
                },
            ],
            # One running copy handles every document; without this a drop can
            # still spawn a second instance.
            "LSMultipleInstancesProhibited": True,
        },
    )
